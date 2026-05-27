'''
This script is a baseline model without the token order component, 
which is the same as the original GeoBloom model.
It is used to compare with geobloom_w_order.py.
'''

import torch
import jieba_fast
import torch_sparse
import numpy as np
import torch.optim as optim
import torch.utils.data

from torch import nn
from tqdm import tqdm


from lambdarank import lambdaLoss
from torch_sparse.tensor import SparseTensor
from bloom_filter import load_data

from nltk.tokenize import word_tokenize
import nltk

dataset = 'GeoGLUE_clean_subset'

def custom_ndcg(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    if len(truth) == 0:
        return 0
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

def recall(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    if len(truth) == 0:
        return 0
    return len(set(prediction[:k]).intersection(set(truth))) / min(len(set(truth)), k)

def eval_search(top_k_indices, query_truth, metrics=None):
    if metrics is None:
        metrics = {
            'Recall @ 20': lambda p, t: recall(p, t, 20),
            'Recall @ 10': lambda p, t: recall(p, t, 10),
            'NDCG @ 5': lambda p, t: custom_ndcg(p, t, 5),
            'NDCG @ 1': lambda p, t: custom_ndcg(p, t, 1),
        }

    results = {}
    for metric_name, metric_func in metrics.items():
        metric_value = np.mean([metric_func(p, t) for p, t in zip(top_k_indices, query_truth)])
        results[metric_name] = metric_value
    return results

# Download required resources
nltk.download('punkt')
if dataset == 'Synthetic':
    tokenizer = lambda x: word_tokenize(x)
else:
    tokenizer = lambda x: jieba_fast.lcut(x)

batch_size = 16
poi_bloom_filter_list, poi_locations, poi_truth = load_data(f'data/{dataset}/poi.txt', 2, 4096, is_query=False, query_tokenizer=tokenizer, poi_tokenizer=tokenizer)
poi_locations = torch.tensor(poi_locations, dtype=torch.float32, device='cuda')

poi_row = []
poi_col = []
poi_values = []
for i in range(len(poi_bloom_filter_list)):
    col_list = list(poi_bloom_filter_list[i])
    # sort the col_list
    col_list.sort()
    poi_col.extend(col_list)
    poi_row.extend([i] * len(col_list))

poi_row = torch.tensor(poi_row, dtype=torch.long, device='cuda')
poi_col = torch.tensor(poi_col, dtype=torch.long, device='cuda')
poi_len = len(poi_bloom_filter_list)
poi_bloom_filter = SparseTensor(row=poi_row, col=poi_col, sparse_sizes=(poi_len, 8192))

# prepare the query bloom filters to evaluate retrieval tasks

def prepare_queries(query_path):
    query_bloom_filter_list, query_locations, query_truth = load_data(query_path, 2, 4096, is_query=True, query_tokenizer=tokenizer, poi_tokenizer=tokenizer)

    query_col = []
    query_row = []
    for i in range(len(query_bloom_filter_list)):
        col_list = list(query_bloom_filter_list[i])
        # sort the col_list
        col_list.sort()
        query_col.extend(col_list)
        query_row.extend([i] * len(col_list))

    query_row = torch.tensor(query_row, dtype=torch.long, device='cuda')
    query_col = torch.tensor(query_col, dtype=torch.long, device='cuda')
    query_bloom_filter = SparseTensor(row=query_row, col=query_col, sparse_sizes=(len(query_bloom_filter_list), 8192))
    return query_bloom_filter, query_locations, query_truth

test_path = f'data/{dataset}/test.txt'
test_bloom_filter, test_locations, test_truth = prepare_queries(test_path)

train_path = f'data/{dataset}/train.txt'
train_bloom_filter, train_locations, train_truth = prepare_queries(train_path)

dev_path = f'data/{dataset}/dev.txt'
dev_bloom_filter, dev_locations, dev_truth = prepare_queries(dev_path)

class SimpleGeoBloom(nn.Module):
    def __init__(self, in_channels, hidden_1, hidden_2, out_channels):
        super(SimpleGeoBloom, self).__init__()
        self.encoder = nn.Linear(in_channels, hidden_1)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, out_channels, bias=False),
        )

        self.decoder[-1].weight.data.fill_(0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=1/8)

        # Distance Modeling
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
        self.d = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))


    def cross_encode(self, query_x: SparseTensor, query_loc: torch.Tensor, poi_x: torch.Tensor, poi_loc: torch.Tensor):
        with torch.no_grad():
            query_dense = torch.zeros(query_x.sparse_size(0), query_x.sparse_size(1), dtype=torch.float32, device='cuda')
            query_row, query_col, _ = query_x.coo()
            query_dense[query_row, query_col] = 1
            poi_x_dense = poi_x
            query_bits = query_dense.sum(dim=-1, keepdim=True)
            poi_bits = poi_x_dense.sum(dim=-1, keepdim=True)
            interaction = torch.mul(query_dense.unsqueeze(1), poi_x_dense.unsqueeze(0))

        query_embed = self.encoder(query_dense) / query_bits
        poi_embed = self.encoder(poi_x_dense) / poi_bits
        poi_embed = poi_embed.unsqueeze(0).expand(query_embed.shape[0], -1, -1)
        all_embed = torch.cat([query_embed.unsqueeze(1).expand(-1, poi_embed.shape[1], -1), poi_embed], dim=-1)
        weights = self.leaky_relu(self.decoder(all_embed)) + 1
        desc_sim = (interaction * weights).sum(dim=-1) / query_bits
        # return desc_sim
        dist = torch.sum((query_loc.unsqueeze(1) - poi_loc) ** 2, dim=-1).sqrt() # [B, K]
        
        
        # Normalization and Score Fusion
        desc_std, desc_mean = torch.std_mean(desc_sim, dim=-1, keepdim=True)
        desc_sim = (desc_sim - desc_mean) / (desc_std + 1e-6)
        dist_sim = - torch.log(dist + 1)
        
        relevance_score = (self.c - (self.a * desc_sim + self.b).sigmoid()) * (dist_sim - self.d)
        return relevance_score



class ListwiseDataset(torch.utils.data.Dataset):
    def __init__(self, train_bloom_filter, train_locations, train_truth):
        self.train_bloom_filter = train_bloom_filter
        self.train_locations = torch.tensor(train_locations, dtype=torch.float32, device='cuda')
        self.train_truth = train_truth

    def __len__(self):
        return self.train_bloom_filter.sparse_size(0)

    def __getitem__(self, idx):
        return idx, self.train_locations[idx], self.train_truth[idx]
    
    def collate_fn(self, batch):
        # select the query via the torch_sparse.index_select
        indices, query_locs, candidate_truths = zip(*batch)
        query = torch_sparse.index_select(self.train_bloom_filter, 0, torch.tensor(indices, dtype=torch.long, device='cuda'))
        query_locs = torch.vstack(query_locs)
        # candidate_truths is a list of [batch_size, 1]
        # we turn it to one-hot encoding
        truths = torch.zeros(len(indices), poi_len, dtype=torch.float32, device='cuda')
        cols = torch.tensor(candidate_truths, dtype=torch.long, device='cuda').squeeze(1)
        truths[torch.arange(len(indices)), cols] = 1
        return query, query_locs, truths
    
    
# Define hyperparameters
learning_rate = 5e-3
num_epochs = 10
hidden_1 = 128
hidden_2 = 32
out_channels = 8192

# Initialize the model, loss function, and optimizer
model = SimpleGeoBloom(in_channels=8192, hidden_1=hidden_1, hidden_2=hidden_2, out_channels=out_channels).to('cuda')
criterion = lambda pred, truth: lambdaLoss(pred, truth, k=10, reduction="sum")
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Prepare the training dataset and dataloader
train_dataset = ListwiseDataset(train_bloom_filter, train_locations, train_truth)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

# Prepare the development dataset and dataloader
dev_dataset = ListwiseDataset(dev_bloom_filter, dev_locations, dev_truth)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn)

# Prepare the test dataset and dataloader
test_dataset = ListwiseDataset(test_bloom_filter, test_locations, test_truth)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

poi_bloom_filter = poi_bloom_filter.to_dense()


model.eval()
with torch.no_grad():
    test_relevance_scores = []
    test_candidate_truths = []
    test_pbar = tqdm(test_loader, desc='Inference on Test Set')
    for batch in test_pbar:
        query, query_loc, candidate_truths = batch
        relevance_scores = model.cross_encode(query, query_loc, poi_bloom_filter, poi_locations)
        test_relevance_scores.append(relevance_scores)
        for i in range(candidate_truths.shape[0]):
            test_candidate_truths.append(candidate_truths[i].nonzero().squeeze(1).cpu().numpy().tolist())
    test_relevance_scores = torch.cat(test_relevance_scores, dim=0)
    test_topk = torch.topk(test_relevance_scores, test_relevance_scores.shape[1], largest=True)[1].cpu().numpy()
    test_results = eval_search(test_topk, test_candidate_truths)
    print(test_results)

# Training loop
best_dev = 0
for epoch in range(num_epochs):

    dev_relevance_scores = []
    dev_candidate_truths = []
    model.eval()
    with torch.no_grad():
        debug_pbar = tqdm(dev_loader, desc='Inference on Development Set')
        for batch in debug_pbar:
            query, query_loc, candidate_truths = batch
            relevance_scores = model.cross_encode(query, query_loc, poi_bloom_filter, poi_locations)
            dev_relevance_scores.append(relevance_scores)
            for i in range(candidate_truths.shape[0]):
                dev_candidate_truths.append(candidate_truths[i].nonzero().squeeze(1).cpu().numpy().tolist())
    dev_relevance_scores = torch.cat(dev_relevance_scores, dim=0)
    dev_topk = torch.topk(dev_relevance_scores, dev_relevance_scores.shape[1], largest=True)[1].cpu().numpy()
    dev_results = eval_search(dev_topk, dev_candidate_truths)
    search_ndcg = dev_results['NDCG @ 5']
    if search_ndcg > best_dev:
        best_dev = search_ndcg
        torch.save(model.state_dict(), f'ckpt/token_order.ckpt')
    print(dev_results)

    model.train()
    train_relevance_scores = []
    train_candidate_truths = []
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in pbar:
        query, query_loc, candidate_truths = batch
        # Forward pass
        relevance_scores = model.cross_encode(query, query_loc, poi_bloom_filter, poi_locations)
        train_relevance_scores.append(relevance_scores.detach())
        for i in range(candidate_truths.shape[0]):
            train_candidate_truths.append(candidate_truths[i].nonzero().squeeze(1).cpu().numpy().tolist())
        loss = criterion(relevance_scores, candidate_truths)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    train_relevance_scores = torch.cat(train_relevance_scores, dim=0)
    train_topk = torch.topk(train_relevance_scores, train_relevance_scores.shape[1], largest=True)[1].cpu().numpy()
    train_results = eval_search(train_topk, train_candidate_truths)
    print(train_results)


# load the best model
print(f'Loading the best model from ckpt/token_order.ckpt, best NDCG@5: {best_dev}')
model.load_state_dict(torch.load(f'ckpt/token_order.ckpt'))

# evaluate the model
model.eval()
with torch.no_grad():
    test_relevance_scores = []
    test_candidate_truths = []
    test_pbar = tqdm(test_loader, desc='Inference on Test Set')
    for batch in test_pbar:
        query, query_loc, candidate_truths = batch
        relevance_scores = model.cross_encode(query, query_loc, poi_bloom_filter, poi_locations)
        test_relevance_scores.append(relevance_scores)
        for i in range(candidate_truths.shape[0]):
            test_candidate_truths.append(candidate_truths[i].nonzero().squeeze(1).cpu().numpy().tolist())
    test_relevance_scores = torch.cat(test_relevance_scores, dim=0)
    test_topk = torch.topk(test_relevance_scores, test_relevance_scores.shape[1], largest=True)[1].cpu().numpy()
    test_results = eval_search(test_topk, test_candidate_truths)
    print(test_results)

'''
Without Token Order:
On Synthetic Data:
{'Recall @ 20': 0.825, 'Recall @ 10': 0.71, 'NDCG @ 5': 0.5353262335693657, 'NDCG @ 1': 0.45}
On Beijing Subset:
{'Recall @ 20': 0.705, 'Recall @ 10': 0.64, 'NDCG @ 5': 0.4624616755935459, 'NDCG @ 1': 0.36}
On GeoGLUE Subset:
{'Recall @ 20': 0.795, 'Recall @ 10': 0.71, 'NDCG @ 5': 0.43503007075914096, 'NDCG @ 1': 0.25}
'''
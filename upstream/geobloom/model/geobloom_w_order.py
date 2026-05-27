'''
This script tests the optional token order component of the model.
It simply adds a convolution module with residual connection to the original GeoBloom model.
The input bloom filter is modified to be [L, 2*4096], where L is the sequence length.
'''

import torch
import numpy as np
import torch.optim as optim
import torch.utils.data
import jieba_fast

from torch import nn
from tqdm import tqdm
from typing import List, Set


from lambdarank import lambdaLoss
from bloom_filter import make_hashfuncs

dataset = 'GeoGLUE_clean_subset'

if dataset == 'Synthetic':
    from nltk.tokenize import word_tokenize
    import nltk

    # Download required resources
    nltk.download('punkt')
    tokenizer = lambda x: word_tokenize(x)
else:
    tokenizer = lambda x: jieba_fast.lcut(x)


def load_data(file_dir, num_slices, num_bits, is_query=True):
    '''
    This function considers the token order.
    bloom_filters: [
        [hash_set_1, hash_set_2, ...],
        [hash_set_1, hash_set_2, ...],
        ...
    ]
    '''
    bloom_filters: List[List[Set[int]]] = []
    locs = []
    truths = []

    hash_func_inner, _ = make_hashfuncs(num_slices, num_bits)

    def hash_func(t):
        hash_list = list(hash_func_inner(t))
        for i in range(1, num_slices):
            hash_list[i] += i * num_bits
        return set(hash_list)

    with open(file_dir, 'r',) as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Loading '+ ('query' if is_query else 'POI') + ' data'):
            line = line.strip().lower().split('\t')
            text = tokenizer(line[0])
            bloom_filter = []
            for t in text:
                bloom_filter.append(hash_func(t))
                if len(bloom_filter) >= 40:
                    break
            bloom_filters.append(bloom_filter)
            x, y = float(line[1]), float(line[2])
            locs.append([x, y])
            if is_query:
                truths.append([int(x) for x in line[3].split(',')])
    return bloom_filters, locs, truths


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


batch_size = 16
poi_bloom_filter_list, poi_locations, poi_truth = load_data(f'data/{dataset}/poi.txt', 2, 4096, is_query=False)
poi_len = len(poi_bloom_filter_list)
max_seq_len = max([len(bloom_filter) for bloom_filter in poi_bloom_filter_list])
poi_bloom_filter = torch.zeros(len(poi_bloom_filter_list), max_seq_len, 2*4096, dtype=torch.int16)
poi_locations = torch.tensor(poi_locations, dtype=torch.float32, device='cuda')
for i in tqdm(range(len(poi_bloom_filter_list)), desc='Constructing POI Bloom Filter'):
    for j in range(len(poi_bloom_filter_list[i])):
        bits_set = poi_bloom_filter_list[i][j]
        for bit in bits_set:
            poi_bloom_filter[i, j, bit] = 1

poi_bloom_filter = poi_bloom_filter.cuda().float()

# prepare the query bloom filters to evaluate retrieval tasks

def prepare_queries(query_path):
    query_bloom_filter_list, query_locations, query_truth = load_data(query_path, 2, 4096, is_query=True)
    max_seq_len = max([len(bloom_filter) for bloom_filter in query_bloom_filter_list])
    query_bloom_filter = torch.zeros(len(query_bloom_filter_list), max_seq_len, 2*4096, dtype=torch.int16)
    for i in tqdm(range(len(query_bloom_filter_list)), desc='Constructing Query Bloom Filter'):
        for j in range(len(query_bloom_filter_list[i])):
            bits_set = query_bloom_filter_list[i][j]
            for bit in bits_set:
                query_bloom_filter[i, j, bit] = 1
    query_bloom_filter = query_bloom_filter.cuda().float()
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
        self.encoder = nn.Linear(in_channels, hidden_1, bias=False)
        # The optional convolution module that can be used to capture the token order.
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_1, hidden_1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_1, hidden_1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_1, hidden_1, kernel_size=5, padding=2),
            nn.ReLU(),
        )
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


    def cross_encode(self, query_x: torch.Tensor, query_loc: torch.Tensor, poi_x: torch.Tensor, poi_loc: torch.Tensor):
        # query_x: [B, L, 2*4096]
        # poi_x: [K, L, 2*4096]
        query_hidden = self.encoder(query_x)
        # do convolution along the sequence length
        query_hidden = self.convs(query_hidden.permute(0, 2, 1)).permute(0, 2, 1) + query_hidden
        poi_hidden = self.encoder(poi_x)
        poi_hidden = self.convs(poi_hidden.permute(0, 2, 1)).permute(0, 2, 1) + poi_hidden
        query_embed = query_hidden.mean(dim=1)
        poi_embed = poi_hidden.mean(dim=1)

        query_bloom_filter = torch.sum(query_x, dim=1).clamp(max=1) # [B, 2*4096]
        poi_bloom_filter = torch.sum(poi_x, dim=1).clamp(max=1) # [K, 2*4096]
        interaction = torch.mul(query_bloom_filter.unsqueeze(1), poi_bloom_filter.unsqueeze(0)) # [B, K, 2*4096]
        query_bits = query_bloom_filter.sum(dim=-1, keepdim=True) # [B, 1]

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
        return self.train_bloom_filter.shape[0]

    def __getitem__(self, idx):
        return self.train_bloom_filter[idx], self.train_locations[idx], self.train_truth[idx]
    
    def collate_fn(self, batch):
        # select the query via the torch_sparse.index_select
        queries, query_locs, truths = zip(*batch)
        query = torch.stack(queries)
        query_locs = torch.vstack(query_locs)
        # candidate_truths is a list of [batch_size, 1]
        # we turn it to one-hot encoding
        target = torch.zeros(len(truths), poi_len, dtype=torch.float32, device='cuda')
        cols = torch.tensor(truths, dtype=torch.long, device='cuda').squeeze(1)
        target[torch.arange(len(truths)), cols] = 1
        return query, query_locs, target
    
    
# Define hyperparameters
learning_rate = 5e-3
num_epochs = 10
hidden_1 = 256
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
print(f'Loading the best model from ckpt/token_order.ckpt, best NDCG@5ß: {best_dev}')
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
With Token Order:
On Synthetic Data:
{'Recall @ 20': 0.985, 'Recall @ 10': 0.985, 'NDCG @ 5': 0.9407824118439408, 'NDCG @ 1': 0.875}
On Beijing Subset:
{'Recall @ 20': 0.705, 'Recall @ 10': 0.63, 'NDCG @ 5': 0.4533354431155974, 'NDCG @ 1': 0.345}
On GeoGLUE Subset:
{'Recall @ 20': 0.775, 'Recall @ 10': 0.7, 'NDCG @ 5': 0.43327996559417004, 'NDCG @ 1': 0.245}
'''
import torch
import numpy as np

from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from torch.utils.data import DataLoader, TensorDataset

MODEL_NAME = "bert-base-chinese"
BATCH_SIZE = 8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GeoGLUE_clean', help='dataset name')

dataset = parser.parse_args().dataset
poi_lines = None
poi_texts = []
with open(f'data/{dataset}/poi.txt', 'r', encoding='utf-8') as file:
    poi_lines = file.readlines()
    print(f'Loaded {len(poi_lines)} lines from poi.txt')
    for poi_line in poi_lines:
        poi_line = poi_line.strip()
        fields = poi_line.split('\t')
        assert len(fields) == 3
        if 'Beijing' in dataset or 'Shanghai' in dataset:
            # We ignore address for the two Meituan datasets, otherwise the results are too bad
            components = fields[0].split(',')
            poi_texts.append(components[0])
        else:
            poi_texts.append(fields[0])



test_queries = []
with open(f'data/{dataset}/test.txt', 'r', encoding='utf-8') as file:
    test_lines = file.readlines()
    print(f'Loaded {len(test_lines)} lines from test.txt')
    for test_line in test_lines:
        test_line = test_line.strip()
        fields = test_line.split('\t')
        assert len(fields) == 4
        test_queries.append(fields[0])

embedding_path = f'baselines/BERT/embeddings/{dataset}'
import os
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

# Compute the embeddings for the POI txt

bert = BertModel.from_pretrained(MODEL_NAME).to('cuda')
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# Function to encode texts using the tokenizer
def encode_texts(texts, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Text tokenization"):
        encoded_dict = tokenizer.encode_plus(
            text,                      
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,    # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attention masks.
            return_tensors='pt',     # Return pytorch tensors.
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to generate embeddings
def generate_embeddings(model, dataloader):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            input_ids, attention_mask = batch
            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Take the embeddings from the last hidden state
            # You might want to experiment with pooling strategies (e.g., mean pooling)
            # Here, we simply take the embedding of the [CLS] token (first token)
            cls_embeddings = mean_pooling(outputs, attention_mask)
            cls_embeddings = cls_embeddings.cpu().numpy()
            embeddings.append(cls_embeddings)

    embeddings = np.vstack(embeddings)
    return embeddings

# Tokenize POI texts and test queries
poi_input_ids, poi_attention_masks = encode_texts(poi_texts, tokenizer)
test_input_ids, test_attention_masks = encode_texts(test_queries, tokenizer)

# Create DataLoader for POIs and test queries
poi_dataset = TensorDataset(poi_input_ids, poi_attention_masks)
test_dataset = TensorDataset(test_input_ids, test_attention_masks)

poi_dataloader = DataLoader(poi_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Generate embeddings
poi_embeddings = generate_embeddings(bert, poi_dataloader)
np.save(os.path.join(embedding_path, 'poi_embeddings.npy'), poi_embeddings)
test_embeddings = generate_embeddings(bert, test_dataloader)
np.save(os.path.join(embedding_path, 'test_embeddings.npy'), test_embeddings)


print("Embeddings generated and saved successfully.")
import os
import numpy as np
import argparse
from tqdm import tqdm


from openai import OpenAI
# Release note: Please provide your openai key here if you want to reproduce the results
# The embeddings in the paper is produced on March 2024.
# It will cost $1 dollar.

# Set the OpenAI API key
if os.path.exists('openai_api_key.txt'):
    with open('openai_api_key.txt', 'r', encoding='utf-8') as f:
        OPENAI_API_KEY = f.read().strip()
else:
    OPENAI_API_KEY = None

assert OPENAI_API_KEY is not None
client = OpenAI(api_key=OPENAI_API_KEY)

# Get the embeddings from OpenAI's text-embedding-small
def get_embedding(text, model="text-embedding-3-small"):
   return client.embeddings.create(input = text, model=model)

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

embedding_path = f'baselines/OpenAI/embeddings/{dataset}'

if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

poi_embedding_path = f'{embedding_path}/poi_embeddings.npy'
test_embedding_path = f'{embedding_path}/test_embeddings.npy'

batch_size = 100
poi_embeddings = []
for i in tqdm(range(0, len(poi_texts), batch_size), desc="Getting POI embeddings"):
    batch = poi_texts[i:i+batch_size]
    response = get_embedding(batch)
    embeddings = [response.data[i].embedding for i in range(len(batch))]
    poi_embeddings.extend(embeddings)

poi_embeddings = np.array(poi_embeddings)
np.save(poi_embedding_path, poi_embeddings)
print(f'Saved POI embeddings to {poi_embedding_path}')

test_embeddings = []
for i in tqdm(range(0, len(test_queries), batch_size), desc="Getting test embeddings"):
    batch = test_queries[i:i+batch_size]
    response = get_embedding(batch)
    embeddings = [response.data[i].embedding for i in range(len(batch))]
    test_embeddings.extend(embeddings)

test_embeddings = np.array(test_embeddings)
np.save(test_embedding_path, test_embeddings)
print(f'Saved test embeddings to {test_embedding_path}')






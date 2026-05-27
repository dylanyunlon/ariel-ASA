'''
This script creates a small subset of the large dataset to test the token order module.
Theoretically, incorporating token order should help the model to distinguish between some of queries and POIs that confuse BM25. Hence, we randomly sample 5000 queries from the large dataset to form the training set, use the top-10 BM25 to form as the POIs.
'''

import random
import os

import numpy as np

# set random seed
random.seed(42)

dataset = 'GeoGLUE_clean'
output_dataset = f'{dataset}_subset'

poi_path = f'data/{dataset}/poi.txt'
test_path = f'data/{dataset}/test.txt'
neg_path = f'result/{dataset}_BM25_D_top100.npy'

with open(poi_path, 'r', encoding='utf-8') as f:
    poi_lines = f.readlines()

test_queries = []


with open(test_path, 'r', encoding='utf-8') as f:
    test_lines = f.readlines()
    query_id = 0
    for line in test_lines:
        query, lat, lon, truth = line.strip().split('\t')
        truth = truth.split(',')
        if len(truth) == 1:
            test_queries.append((query_id, query, lat, lon, int(truth[0])))
        query_id += 1

test_negative = np.load(neg_path)

# randomly sample 1000 test queries
sampled_test_queries = random.sample(test_queries, 1000)

new_queries = []
new_poi_lines = []

for test_query in sampled_test_queries:
    query_id, query, lat, lon, truth = test_query
    test_negative_idx = test_negative[query_id]
    new_truth_id = len(new_poi_lines)
    new_queries.append(query + '\t' + lat + '\t' + lon + '\t' + str(new_truth_id) + '\n')
    new_poi_lines.append(poi_lines[truth])
    count = 0
    for neg_idx in test_negative_idx:
        if neg_idx == truth:
            continue
        if count >= 4:
            break
        count += 1
        new_poi_lines.append(poi_lines[neg_idx])
    

random.shuffle(new_queries)
new_train_lines = new_queries[:700]
new_dev_lines = new_queries[700:800]
new_test_lines = new_queries[800:]

print(f'Number of queries: {len(new_queries)}')

os.makedirs(f'data/{output_dataset}', exist_ok=True)

with open(f'data/{output_dataset}/poi.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_poi_lines)

with open(f'data/{output_dataset}/train.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_train_lines)

with open(f'data/{output_dataset}/dev.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_dev_lines)

with open(f'data/{output_dataset}/test.txt', 'w', encoding='utf-8') as f:
    f.writelines(new_test_lines)


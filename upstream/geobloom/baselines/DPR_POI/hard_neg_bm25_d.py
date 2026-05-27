'''
    This script reuse the BM25-D method to generate hard negatives for the DPR model.
'''
import os
import jieba_fast
import torch

import numpy as np

from tqdm import tqdm

min_x = 1e9
max_x = -1e9
min_y = 1e9
max_y = -1e9

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GeoGLUE_clean')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--portion', type=str, default='1')


dataset = parser.parse_args().dataset
print (dataset)
split = parser.parse_args().split
portion = parser.parse_args().portion
portion = portion if portion != '1' and split == 'train' else None

processed_path = f'data/{dataset}'
processed_poi_file = processed_path + '/poi.txt'
processed_query_file = processed_path + '/' + parser.parse_args().split + '.txt' if portion is None else processed_path + '/portion/' + split + f'_{portion}.txt'

query_txt = []
query_locations = []
query_truth = []

with open(processed_query_file, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        query_txt.append(line[0])
        query_utm_lat = float(line[1])
        query_utm_lon = float(line[2])
        query_locations.append([query_utm_lat, query_utm_lon])
        query_truth_str = line[3]
        query_truth_split = query_truth_str.split(',')
        query_truth.append([int(x) for x in query_truth_split])

poi_txt = []
poi_locations = []
# get the max and min of the utm coordinates
with open(processed_poi_file, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc='Splitting input text'):
        line = line.strip().split('\t')
        poi_txt.append(jieba_fast.lcut_for_search(line[0]))
        poi_utm_lat = float(line[1])
        poi_utm_lon = float(line[2])
        poi_locations.append([poi_utm_lat, poi_utm_lon])
        min_x = min(min_x, poi_utm_lat)
        max_x = max(max_x, poi_utm_lat)
        min_y = min(min_y, poi_utm_lon)
        max_y = max(max_y, poi_utm_lon)

poi_locations = np.array(poi_locations)
d_norm = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

# process the bm25 with threadpool
from concurrent.futures import ProcessPoolExecutor, as_completed
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(poi_txt, k1=0.3, b=0.1)

def bm25_search(query):
    tokenized_query = jieba_fast.lcut_for_search(query)
    doc_scores = bm25.get_scores(tokenized_query)
    return doc_scores.astype(np.uint8)

all_scores = [None] * len(query_txt)

with ProcessPoolExecutor(max_workers=16) as executor:
    queries = [None] * len(query_txt)

    futures_to_index = {executor.submit(bm25_search, query): idx for idx, query in enumerate(query_txt)}

    # Display progress bar and collect results
    for future in tqdm(as_completed(futures_to_index), total=len(queries), desc="Processing queries"):
        scores = future.result()
        index = futures_to_index[future]
        all_scores[index] = scores

    # most of the bm25 scores are 0. We can save the bm25 scores to a sparse matrix
    # and use it to speed up the reranking process

query_locations = np.array(query_locations)
poi_locations = np.array(poi_locations)

query_locations = torch.from_numpy(query_locations).cuda()
poi_locations = torch.from_numpy(poi_locations).cuda()


top_indice_100_list = []

# Now for each query, we search for the top-k nearest POIs and rerank them
for index, query in enumerate(tqdm(query_txt)):
    # The predict score is 𝑆𝑇(𝑝,𝑞) =(1−𝛼)×(1−𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐))+ 𝛼 ×𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐)
    # where 𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐) is the distance similarity between the location of POI p and the location of query q
    # and 𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐) is the bm25 score between the description of POI p and the description of query q.
    # and 𝛼 is a hyper-parameter to balance the two parts (we set 𝛼 = 0.1 in our experiments).
    alpha = 0.4
    query_loc = query_locations[index]
    # First compute the distance similarity
    dist_sim = torch.sum((poi_locations - query_loc) ** 2, dim=1).sqrt()
    dist_sim = 1 - dist_sim / d_norm
    # Then compute the description similarity
    desc_sim = torch.from_numpy(all_scores[index]).cuda().float()
    # desc_sim = desc_sim
    # Combine the two
    predict_score = (1 - alpha) * dist_sim + alpha * desc_sim
    # write the result to the matrix
    #predict_result_matrix[query_map[query]] = predict_score
    # Get the top-10 indices
    top_indices = torch.topk(predict_score, 100, largest=True)[1].cpu().numpy()
    top_indice_100_list.append(top_indices)

# Save the top-100 indices
path = f'baselines/DPR_POI/hard_negatives/{dataset}/portion' if portion is not None else f'baselines/DPR_POI/hard_negatives/{dataset}'
file_name = f'{split}_{portion}.npy' if portion is not None else f'{split}.npy'
if not os.path.exists(path):
    os.makedirs(path)
np.save(f'{path}/{file_name}', np.vstack(top_indice_100_list).astype(np.uint32))
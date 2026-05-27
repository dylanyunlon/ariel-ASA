'''
    This is a simple baseline that test the performance of bm25.
    It compute the query score to all the POIs and save the scores as a sparse matrix.
    The next time of calling we just load the matrix and do the ranking.
'''

import jieba_fast

import numpy as np

from tqdm import tqdm

def fast_ndcg(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GeoGLUE_clean')

dataset = parser.parse_args().dataset
processed_path = f'data/{dataset}'
processed_poi_file = processed_path + '/poi.txt'
processed_query_file = processed_path + '/test.txt'

query_txt = []
query_truth = []

with open(processed_query_file, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc='Loading query text'):
        line = line.strip().split('\t')
        line = line.strip().split('\t')
        query_txt.append(line[0])
        query_truth_str = line[3]
        query_truth_split = query_truth_str.split(',')
        query_truth.append([int(x) for x in query_truth_split])

poi_txt = []
# get the max and min of the utm coordinates
with open(processed_poi_file, 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines, desc='Splitting POI text'):
        line = line.strip().split('\t')
        line = line.strip().split('\t')
        poi_txt.append(jieba_fast.lcut_for_search(line[0]))


# process the bm25 with threadpool
from concurrent.futures import ProcessPoolExecutor, as_completed
from rank_bm25 import BM25Okapi

# It is crucially important to set the parameters of BM25Okapi
# otherwise the performance will be very bad
bm25 = BM25Okapi(poi_txt, k1=0.3, b=0.1)

def bm25_search(query):
    tokenized_query = jieba_fast.lcut_for_search(query)
    doc_scores = bm25.get_scores(tokenized_query)
    # return top100 indices
    top_indices = np.argsort(doc_scores)[::-1][:100]
    return top_indices.astype(np.uint32)

all_scores = [None] * len(query_txt)

with ProcessPoolExecutor(max_workers=32) as executor:
    queries = [None] * len(query_txt)

    futures_to_index = {executor.submit(bm25_search, query): idx for idx, query in enumerate(query_txt)}

    # Display progress bar and collect results
    for future in tqdm(as_completed(futures_to_index), total=len(queries), desc="Processing queries"):
        scores = future.result()
        index = futures_to_index[future]
        all_scores[index] = scores


recalls_1 = []
recalls_5 = []
recalls_20 = []
recalls_50 = []

top_indice_100_list = []

# Now for each query, we search for the top-k nearest POIs and rerank them
for index, query in enumerate(tqdm(query_txt)):
    # The predict score is 𝑆𝑇(𝑝,𝑞) =(1−𝛼)×(1−𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐))+ 𝛼 ×𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐)
    # where 𝑆𝐷𝑖𝑠𝑡(𝑝.𝑙𝑜𝑐,𝑞.𝑙𝑜𝑐) is the distance similarity between the location of POI p and the location of query q
    # and 𝑇𝑅𝑒𝑙(𝑝.𝑑𝑜𝑐,𝑞.𝑑𝑜𝑐) is the bm25 score between the description of POI p and the description of query q.
    # and 𝛼 is a hyper-parameter to balance the two parts (we set 𝛼 = 0.1 in our experiments).
    top_indices = all_scores[index]
    top_indice_100_list.append(top_indices)

    truth = set(query_truth[index])

    recall_1 = 1 if top_indices[0] in truth else 0
    recall_5 = len(set(top_indices[:5]) & truth) / min(5, len(truth))
    recall_20 = len(set(top_indices[:20]) & truth) / min(20, len(truth))
    recall_50 = len(set(top_indices[:50]) & truth) / min(50, len(truth))

    recalls_1.append(recall_1)
    recalls_5.append(recall_5)
    recalls_20.append(recall_20)
    recalls_50.append(recall_50)


# Calculate the mean recall@5 and recall@10
print('recall@1', np.mean(recalls_1))
print('recall@5', np.mean(recalls_5))
print('recall@20', np.mean(recalls_20))
print('recall@50', np.mean(recalls_50))

# Save the top-100 indices
np.save('result/{}_BM25_top100.npy'.format(dataset), top_indice_100_list)
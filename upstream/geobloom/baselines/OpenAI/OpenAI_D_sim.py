import argparse
import numpy as np
import torch

from tqdm import trange


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GeoGLUE_clean')

dataset = parser.parse_args().dataset
processed_path = f'data/{dataset}'


poi_embeddings = np.load(f'baselines/OpenAI/embeddings/{dataset}/poi_embeddings.npy')
test_embeddings = np.load(f'baselines/OpenAI/embeddings/{dataset}/test_embeddings.npy')

# for each test embedding, find the top k poi embeddings with the highest cosine similarit

poi_embeddings = torch.tensor(poi_embeddings, dtype=torch.float16).cuda()
test_embeddings = torch.tensor(test_embeddings, dtype=torch.float16).cuda()


# We also consider the distance. It needs to be loaded from the file, and normalized.

processed_poi_file = processed_path + '/poi.txt'
processed_query_file = processed_path + '/test.txt'


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

min_x = 1e9
max_x = -1e9
min_y = 1e9
max_y = -1e9

with open(processed_poi_file, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        poi_txt.append(line[0])
        poi_utm_lat = float(line[1])
        poi_utm_lon = float(line[2])
        poi_locations.append([poi_utm_lat, poi_utm_lon])
        min_x = min(min_x, poi_utm_lat)
        max_x = max(max_x, poi_utm_lat)
        min_y = min(min_y, poi_utm_lon)
        max_y = max(max_y, poi_utm_lon)


poi_locations = np.array(poi_locations)
d_norm = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

query_locations = np.array(query_locations)
poi_locations = np.array(poi_locations)

query_locations = torch.from_numpy(query_locations).cuda()
poi_locations = torch.from_numpy(poi_locations).cuda()

top_k_indices = []
batch_size = 16384

if 'GeoGLUE' in dataset:
    # Grid search shows that alpha = 0.8 is the best for GeoGLUE queries
    alpha = 0.8
else:
    # Grid search shows that alpha = 0.3 is the best for Meituan queries
    alpha = 0.3


for i in trange(test_embeddings.size(0)):

    test_embedding = test_embeddings[i].unsqueeze(0)
    all_cosine_sims = []

    # Process poi_embeddings in batches
    for batch_start in range(0, poi_embeddings.size(0), batch_size):
        batch_end = min(batch_start + batch_size, poi_embeddings.size(0))
        poi_batch = poi_embeddings[batch_start:batch_end]

        # Compute cosine similarity for the batch
        cosine_sim_batch = torch.cosine_similarity(poi_batch, test_embedding, dim=1)
        all_cosine_sims.append(cosine_sim_batch)

    # Concatenate the results from all batches
    all_cosine_sims = torch.cat(all_cosine_sims, dim=0)
    desc_sim = all_cosine_sims
    # Scale the cosine similarity to [0, 1]
    desc_sim = (desc_sim - desc_sim.min()) / (desc_sim.max() - desc_sim.min())

    query_loc = query_locations[i]
    # First compute the distance similarity
    dist_sim = torch.sum((poi_locations - query_loc) ** 2, dim=1).sqrt()
    dist_sim = 1 - dist_sim / d_norm

    # Combine the two
    predict_score = (1 - alpha) * dist_sim + alpha * desc_sim
    # write the result to the matrix
    #predict_result_matrix[query_map[query]] = predict_score
    # Get the top-10 indices
    top_indices = torch.topk(predict_score, 100, largest=True)[1].cpu().numpy()

    top_k_indices.append(top_indices)


# save it to the result file
top_k_indices = np.array(top_k_indices)
np.save(f'result/{dataset}_OpenAI_D_top100.npy', top_k_indices)
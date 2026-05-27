'''
This script analyzes the outcomes of our GeoBloom.
1. It checks the Recall@20, Recall@10, NDCG@5, and NDCG@1 (sanity check, whether it is the same as the results reported by the NNUE engine).
2. It outputs the top-20 results for each query, and according to the truth, it is divided into three files: success.txt, top20.txt, and fail.txt.
'''
import os
import argparse
import struct
import numpy as np

from tqdm import trange
from evaluation import fast_ndcg, recall, evaluate


def load_text(text_path):
    text = []
    locations = []
    truths = []
    with open(text_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            text.append(line[0])
            utm_lat = float(line[1])
            utm_lon = float(line[2])
            locations.append([utm_lat, utm_lon])
            if len(line) > 3:
                truths.append([int(x) for x in line[3].split(',')])

    return text, locations, truths

def deserialize_tree(dataset_name):
    with open(f'data_bin/{dataset_name}/tree.bin', 'rb') as f:
        num_levels = struct.unpack('I', f.read(4))[0]
        levels = []
        for _ in range(num_levels):
            num_clusters = struct.unpack('I', f.read(4))[0]
            clusters = []
            for _ in range(num_clusters):
                num_nodes = struct.unpack('I', f.read(4))[0]
                cluster = []
                for _ in range(num_nodes):
                    node_id = struct.unpack('I', f.read(4))[0]
                    cluster.append(node_id)
                clusters.append(cluster)
            levels.append(clusters)
        return levels
    
def load_candidates(levels, file_path, num_rows, beam_widths):
    '''
        This function is used to deserialize the candidates from the C++ inference engine.
        The bin file: [query_id][depth][beam_width], unsigned int32
        candidates: [depth][query_id][beam_width]
    '''
    depths = len(levels)
    if not isinstance(beam_widths, list):
        beam_widths = [beam_widths] * (depths)
    candidates = [[None for i in range(num_rows)] for j in range(depths)]
    topk_list = [[] for _ in range(num_rows)]
    with open(file_path, 'rb') as f:
        for query_id in trange(num_rows, desc='Deserializing candidates'):
            for depth in range(depths):
                beam_width = min(beam_widths[depth], len(levels[depth]))
                if depth != depths - 1:
                    # skip the 4 * beam_width bytes
                    f.seek(4 * beam_width, os.SEEK_CUR)
                else:
                    topk_list[query_id] = struct.unpack(f'{beam_width}I', f.read(beam_width * 4))
    return candidates, topk_list

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GeoGLUE')
    args = parser.parse_args()
    dataset = args.dataset
    bin_path = f'data_bin/{dataset}/test_nodes.bin'
    poi_text_path = f'data/{dataset}/poi.txt'
    query_text_path = f'data/{dataset}/test.txt'
    if dataset == 'GeoGLUE':
        num_slices = 2
        num_bits = 16384
        beam_width = 4000
    elif dataset == 'Beijing' or dataset == 'Shanghai':
        num_slices = 2
        num_bits = 8192
        beam_width = 400
    elif dataset == 'GeoGLUE_clean':
        num_slices = 2
        num_bits = 8192
        beam_width = 800

    poi_text, poi_locations, _ = load_text(poi_text_path)
    query_text, query_locations, truths = load_text(query_text_path)
    levels = deserialize_tree(dataset)
    # hard-coded depth = 4
    levels = levels[:3]
    # reverse the levels
    levels = levels[::-1]
    levels.append(poi_text)
    candidates, topk_list = load_candidates(levels, bin_path, len(query_text), beam_width)

    metrics = {
        # 'Recall @ 1000': lambda p, t: recall(p, t, 1000),
        'Recall @ 20': lambda p, t: recall(p, t, 20),
        'Recall @ 10': lambda p, t: recall(p, t, 10),
        'NDCG @ 5': lambda p, t: fast_ndcg(p, t, 5),
        'NDCG @ 1': lambda p, t: fast_ndcg(p, t, 1),
    }

    results = evaluate(topk_list, truths, metrics)
    result_string = ''
    for metric_name, metric_value in results.items():
        result_string += f'{metric_name}: {metric_value}\t'
    print(result_string[:-1])

    # output it to the result/GeoBloom/{dataset}/result.txt
    with open(f'result/GeoBloom/{dataset}/result.txt', 'w') as f:
        f.write(result_string)

    # We divide the cases into three categories:
    # 1. The top-1 is the truth.
    # 2. The top-20 contains the truth.
    # 3. The top-20 does not contain the truth.
    # For case 1, we give the distance between the top-1 and the truth.
    # For case 2, we give all results that ranks before the truth, and the distance between the query and the truth.
    # For case 3, we give the truth, the distance between the query and the truth, and the top-20 results, and the distance.
    # Only keep distances in meters.

    num_queries = len(query_text)
    case1_lines = []
    case2_lines = []
    case3_lines = []

    for i in trange(num_queries):
        query_idx = i
        truth_idx = truths[query_idx][0]
        topk_idx = topk_list[query_idx][:20]
        truth_dist = np.sqrt((query_locations[query_idx][0] - poi_locations[truth_idx][0]) ** 2 + (query_locations[query_idx][1] - poi_locations[truth_idx][1]) ** 2)
        if truth_idx == topk_idx[0]:
            # calculate the euclidean distance between the top-1 and the truth.
            case1_lines.append(f'Query: {query_text[query_idx]}, Truth: {poi_text[truth_idx]}, Distance: {truth_dist:.0f}')
        else:
            try:
                truth_idx_in_topk = topk_idx.index(truth_idx)
            except ValueError:
                truth_idx_in_topk = -1
            if truth_idx_in_topk != -1:
                # The truth is in the top-20.
                case2_lines.append(f'Query: {query_text[query_idx]}, Truth: {poi_text[truth_idx]}, Distance: {truth_dist:.0f}')
                for j in range(truth_idx_in_topk):
                    dist = np.sqrt((query_locations[query_idx][0] - poi_locations[topk_idx[j]][0]) ** 2 + (query_locations[query_idx][1] - poi_locations[topk_idx[j]][1]) ** 2)
                    case2_lines.append(f'\t- POI: {poi_text[topk_idx[j]]}, Distance: {dist:.0f}')
            else:
                # The truth is not in the top-20.
                case3_lines.append(f'Query: {query_text[query_idx]}, Truth: {poi_text[truth_idx]}, Distance: {truth_dist:.0f}')
                for j in range(20):
                    dist = np.sqrt((query_locations[query_idx][0] - poi_locations[topk_idx[j]][0]) ** 2 + (query_locations[query_idx][1] - poi_locations[topk_idx[j]][1]) ** 2)
                    case3_lines.append(f'\t- POI: {poi_text[topk_idx[j]]}, Distance: {dist:.0f}')

    output_dir = f'result/GeoBloom/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/success.txt', 'w') as f:
        for line in case1_lines:
            f.write(line + '\n')
    with open(f'{output_dir}/top20.txt', 'w') as f:
        for line in case2_lines:
            f.write(line + '\n')
    with open(f'{output_dir}/fail.txt', 'w') as f:
        for line in case3_lines:
            f.write(line + '\n')

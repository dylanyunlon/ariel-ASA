import os
import time
import torch
import struct
import jieba_fast

from tqdm import tqdm

from kmeans_tree import build_kmeans_tree
from bloom_filter import load_data

class QueryDataset:
    '''
    The query dataset is used to collate the query data.
    query_bloom_filters: (batch_size, NUM_SLICES * NUM_BITS)
    query_locs: (batch_size, 2)
    truth: (batch_size)

    '''
    def __init__(self, query_bloom_filters, dim, query_locs, truths):
        super().__init__()
        self.max_bit_num = max([len(x) for x in query_bloom_filters])
        self.query_bloom_filters = torch.empty((len(query_bloom_filters), self.max_bit_num), dtype=torch.int16)
        for i, query_bloom_filter in enumerate(query_bloom_filters):
            self.query_bloom_filters[i, :len(query_bloom_filter)] = torch.tensor(query_bloom_filter, dtype=torch.int16)
            self.query_bloom_filters[i, len(query_bloom_filter):] = -1
        self.query_locs = torch.tensor(query_locs, dtype=torch.float32)
        self.dim = dim
        self.truths = truths
    
    def __len__(self):
        return len(self.query_bloom_filters)
    
    def __getitem__(self, idx):
        return self.query_bloom_filters[idx], self.query_locs[idx], self.truths[idx]
    


class POIDataset:
    def __init__(self, dataset: str, num_slices, num_bits, load_query=True, portion=None, preprocess=False):
        super().__init__()
        self.dataset_name: str = dataset
        self.dataset_dir = os.path.join('data', dataset)
        self.data_bin_dir = os.path.join('data_bin', dataset)
        if not os.path.exists(self.data_bin_dir):
            os.makedirs(self.data_bin_dir)
        self.num_slices = num_slices
        self.num_bits = num_bits
        if not preprocess:
            self.poi_bloom_filters, self.poi_locs, _ = self.build_or_load('poi')
            if load_query:
                dim = num_slices * num_bits
                train_bloom_filters, train_locs, train_truths = self.build_or_load('train', portion)
                dev_bloom_filters, dev_locs, dev_truths = self.build_or_load('dev')
                test_bloom_filters, test_locs, test_truths = self.build_or_load('test')
                self.train_dataset = QueryDataset(train_bloom_filters, dim, train_locs, train_truths)
                self.dev_dataset = QueryDataset(dev_bloom_filters, dim, dev_locs, dev_truths)
                self.test_dataset = QueryDataset(test_bloom_filters, dim, test_locs, test_truths)
        else:
            self.build_if_not_exist(portion=portion)

    def build_or_load(self, split, portion=None):
        assert split in ['train', 'dev', 'test', 'poi']
        if portion is not None:
            bin_file = os.path.join(self.data_bin_dir, f'portion/{split}_{portion}.bin')
        else:
            bin_file = os.path.join(self.data_bin_dir, f'{split}.bin')
        if os.path.exists(bin_file):
            bloom_filters, locs, truths = self.deserialize(bin_file)
        else:
            # For GeoGLUE, we use jieba.lcut to avoid the noise issue in the POI data.
            # For other datasets, we use jieba.lcut_for_search.
            raw_file = os.path.join(self.dataset_dir, f'{split}.txt' if portion is None else f'portion/{split}_{portion}.txt')
            bloom_filters, locs, truths = load_data(raw_file, self.num_slices, self.num_bits, is_query=split != 'poi', dict_tokenizer=jieba_fast.lcut if self.dataset_name == 'GeoGLUE' else jieba_fast.lcut_for_search)
            self.serialize(bloom_filters, locs, bin_file, truths)
        return bloom_filters, locs, truths
    
    def build_if_not_exist(self, portion=None):
        for split in ['train', 'dev', 'test', 'poi']:
            if portion is not None:
                bin_file = os.path.join(self.data_bin_dir, f'portion/{split}_{portion}.bin')
            else:
                bin_file = os.path.join(self.data_bin_dir, f'{split}.bin')
            if not os.path.exists(bin_file):
                raw_file = os.path.join(self.dataset_dir, f'{split}.txt' if portion is None else f'portion/{split}_{portion}.txt')
                bloom_filters, locs, truths = load_data(raw_file, self.num_slices, self.num_bits, is_query=split != 'poi')
                self.serialize(bloom_filters, locs, bin_file, truths)
    
    def serialize(self, bloom_filters, locations, file_dir, truths):
        '''
        serialize the bloom filters and locations into a binary file.
        bloom filters are NUM_SLICES * NUM_BITS bits binary, locations are two 32-bit float.
        '''
        num_rows = len(bloom_filters)
        num_cols = self.num_slices * self.num_bits
        if len(truths) > 0:
            assert len(truths) == num_rows
        if not os.path.exists(os.path.dirname(file_dir)):
            os.makedirs(os.path.dirname(file_dir))
        with open(file_dir, 'wb') as file:
            start = time.time()
            file.write(struct.pack('IIH', num_rows, num_cols, 0 if len(truths)==0 else 1))
            # The uint16 is used to store the bloom filter
            # The float64 is used to store the location
            # We try to make the read/write process as fast as possible.
            bloom_filter_lengths = [len(bloom_filter) for bloom_filter in bloom_filters]
            file.write(struct.pack(f'{num_rows}H', *bloom_filter_lengths))
            bloom_filter_data = []
            for bloom_filter in bloom_filters:
                bloom_filter_data.extend(bloom_filter)
            file.write(struct.pack(f'{sum(bloom_filter_lengths)}H', *bloom_filter_data))
            file.write(struct.pack(f'{num_rows * 2}f', *[x for loc in locations for x in loc]))
            if len(truths) > 0:
                for i in range(num_rows):
                    file.write(struct.pack('H', len(truths[i])))
                    for t in truths[i]:
                        file.write(struct.pack('I', t))
            print(f'Serializing {file_dir} takes {time.time() - start} seconds.')

    @staticmethod
    def deserialize(file_dir):
        '''
            deserialize the binary file into bloom filters and locations.
        '''
        with open(file_dir, 'rb') as file:
            start = time.time()
            num_rows, _, has_truth = struct.unpack('IIH', file.read(10))
            bloom_filter_lengths = struct.unpack(f'{num_rows}H', file.read(num_rows * 2))
            bloom_filter_data = struct.unpack(f'{sum(bloom_filter_lengths)}H', file.read(sum(bloom_filter_lengths) * 2))
            bloom_filters = [None] * num_rows
            start_idx = 0
            for row, bloom_filter_length in enumerate(tqdm(bloom_filter_lengths, desc='Reading Bloom filters')):
                bloom_filters[row] = list(bloom_filter_data[start_idx:start_idx + bloom_filter_length])
                # sort the bloom filter by the index
                bloom_filters[row].sort()
                start_idx += bloom_filter_length
            locations = []
            for _ in range(num_rows):
                locations.append(struct.unpack('ff', file.read(8)))
            truths = []
            if has_truth == 1:
                for _ in range(num_rows):
                    num_truths = struct.unpack('H', file.read(2))[0]
                    truths.append(struct.unpack(f'{num_truths}I', file.read(num_truths * 4)))
            print(f'Deserializing {file_dir} takes {time.time() - start} seconds.')
            return bloom_filters, locations, truths


class POIRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, train_bloom_filter, num_slices, num_bits, train_locations, poi_sparse, poi_dense, poi_locations, poi_radius, train_truth, train_neg, top_k):
        self.train_bloom_filter = train_bloom_filter
        self.train_locations = train_locations
        self.poi_sparse: torch.Tensor = poi_sparse
        self.poi_dense: torch.Tensor = poi_dense
        self.poi_locations = poi_locations
        self.poi_radius = poi_radius
        self.num_slices = num_slices
        self.num_bits = num_bits
        self.candidate_poi_ids = []
        self.candidate_truths = []
        self.top_k = top_k
        for i in range(len(train_bloom_filter)):
            truth_id = train_truth[i]
            neg_ids = train_neg[i]
            candidate_poi_id = []
            candidate_truth = []
            for neg_id in neg_ids:
                if len(candidate_poi_id) >= top_k:
                    break
                candidate_poi_id.append(neg_id)
                if neg_id not in truth_id:
                    candidate_truth.append(0)
                else:
                    candidate_truth.append(1)
            assert len(candidate_poi_id) == top_k
            self.candidate_poi_ids.append(candidate_poi_id)
            self.candidate_truths.append(candidate_truth)

        # select the poi bloom filters
        self.candidate_poi_ids = torch.tensor(self.candidate_poi_ids, dtype=torch.long)
        self.candidate_truths = torch.tensor(self.candidate_truths, dtype=torch.float16)

    def __len__(self):
        return len(self.train_bloom_filter)

    
    def __getitem__(self, idx):
        selected_row_idx = self.candidate_poi_ids[idx]
        candidate_poi_sparse = self.poi_sparse[selected_row_idx] if self.poi_sparse is not None else None
        candidate_poi_locs = self.poi_locations[selected_row_idx]
        candidate_poi_radius = self.poi_radius[selected_row_idx]
        candidate_poi_dense = self.poi_dense[selected_row_idx] if self.poi_dense is not None else None

        return self.train_bloom_filter[idx], self.train_locations[idx], candidate_poi_sparse, candidate_poi_dense, candidate_poi_locs, candidate_poi_radius, self.candidate_truths[idx]
    
    def collate_fn(self, batch):
        query_bloom_filters, query_locs, candidate_pois, candidate_poi_dense, candidate_poi_locs, candidate_poi_radius, candidate_truths = zip(*batch)
        query_bloom_filters = torch.stack(query_bloom_filters)
        # The query_bloom_filters is a tensor of shape (batch_size, max_bit_num), but the bits may be significantly less than max_bit_num.
        # We cut the -1 padding in the end.
        query_valid_bits = (query_bloom_filters != -1).sum(dim=-1).max()
        query_bloom_filters = query_bloom_filters[:, :query_valid_bits]
        candidate_poi_dense = torch.stack(candidate_poi_dense) if candidate_poi_dense[0] is not None else None
        query_locs = torch.vstack(query_locs)
        if candidate_pois[0] is not None:
            candidate_pois = torch.stack(candidate_pois)
            candidate_poi_valid_bits = (candidate_pois != -1).sum(dim=-1).max()
            candidate_pois = candidate_pois[:, :, :candidate_poi_valid_bits]
        else:
            candidate_pois = None
        # similarly, the candidate_poi_locs is a tensor of shape (batch_size, K, max_bit_num), but the bits may be significantly less than max_bit_num.
        # We cut the -1 padding in the end.
        candidate_poi_locs = torch.stack(candidate_poi_locs)
        candidate_poi_radius = torch.vstack(candidate_poi_radius)
        candidate_truths = torch.vstack(candidate_truths)
        return query_bloom_filters, query_locs, candidate_pois, candidate_poi_dense, candidate_poi_locs, candidate_poi_radius, candidate_truths
    

    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--portion', type=str, default='1')

    args = parser.parse_args()
    dataset = args.dataset
    portion = args.portion
    portion = None if portion == '1' else portion
    if dataset == 'GeoGLUE': # GeoGLUE has significant more noisy POIs, which requires longer Bloom filters.
        num_slices = 4
        num_bits = 8192
    else:
        num_slices = 2
        num_bits = 8192
    poi_dataset = POIDataset(dataset, num_slices=num_slices, num_bits=num_bits, load_query=True, portion=portion, preprocess=True)

    # building BloomFilterTree in advance.
    if not os.path.exists(f'data_bin/{dataset}/tree.bin'):
        poi_locs = []
        with open(f'data/{dataset}/poi.txt', 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc='Loading POI locations'):
                line = line.strip().split('\t')
                x, y = float(line[1]), float(line[2])
                poi_locs.append([x, y])

        levels = build_kmeans_tree(dataset, poi_locs, width=8)
        with open(f'data_bin/{dataset}/tree.bin', 'wb') as f:
            f.write(struct.pack('I', len(levels)))
            for level in levels:
                f.write(struct.pack('I', len(level)))
                for cluster in level:
                    f.write(struct.pack('I', len(cluster)))
                    for node_id in cluster:
                        f.write(struct.pack('I', node_id))

    
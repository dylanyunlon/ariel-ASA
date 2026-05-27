import os
import math
import torch
import struct

from tqdm import tqdm, trange
from typing import List

from dataset import POIDataset
from kmeans_tree import build_kmeans_tree

class TreeNode:
    def __init__(self, bloom_filter, location, child=None, parent=None, id_in_level=None):
        self.bloom_filter: set = bloom_filter if isinstance(bloom_filter, set) else set(bloom_filter)
        self.location: list = location
        if len(location) == 2:
            self.max_x: float = location[0]
            self.min_x: float = location[0]
            self.max_y: float = location[1]
            self.min_y: float = location[1]
        else:
            self.max_x: float = -1e9
            self.min_x: float = 1e9
            self.max_y: float = -1e9
            self.min_y: float = 1e9

        self.radius: float = 0
        self.child: List[TreeNode] = child
        self.parent: TreeNode = parent
        self.id_in_level: int = id_in_level

    def add_child(self, child):
        self.bloom_filter.update(child.bloom_filter)
        if self.child is None:
            self.child = []
        self.child.append(child)
        child.parent = self
        self.max_x = max(self.max_x, child.max_x)
        self.min_x = min(self.min_x, child.min_x)
        self.max_y = max(self.max_y, child.max_y)
        self.min_y = min(self.min_y, child.min_y)
    
    def compute_radius(self):
        self.radius = math.sqrt((self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2) / 2
        self.location = [(self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2]

    def __repr__(self):
        return f'Node: {self.location}, depth: {self.depth}, radius: {self.radius}, bloom_filter: {len(self.bloom_filter)}'
    

class BloomFilterTree:
    def __init__(self, poi_dataset: POIDataset, width, depth=4) -> None:
        self.dataset_name: str = poi_dataset.dataset_name
        self.bloom_filter_dim: int = poi_dataset.num_slices * poi_dataset.num_bits
        assert self.bloom_filter_dim <= 32768
        self.levels: List[List[TreeNode]] = []
        self.leaf_nodes: List[TreeNode] = []
        self.width: int = width
        self.num_nodes: int = len(poi_dataset.poi_bloom_filters)

        for i in range(len(poi_dataset.poi_bloom_filters)):
            # The id_in_level for leaf nodes is the index of the poi in the poi_dataset
            self.leaf_nodes.append(TreeNode(poi_dataset.poi_bloom_filters[i], poi_dataset.poi_locs[i], id_in_level=i))

        self.levels.insert(0, self.leaf_nodes)

        print('Building the bloom filter tree...')
        if not os.path.exists(f'data_bin/{self.dataset_name}/tree.bin'):
            cluster_by_layers = build_kmeans_tree(self.dataset_name, poi_dataset.poi_locs, width=width)
            self.serialize(cluster_by_layers)
        else:
            cluster_by_layers = self.deserialize()

        prev_level = self.leaf_nodes
        for cluster in cluster_by_layers:
            current_level = self.build_from_clusters(cluster, prev_level)
            self.num_nodes += len(current_level)
            self.levels.insert(0, current_level)
            prev_level = current_level

        self.levels = self.levels[len(self.levels) - depth:]
        self.depth = len(self.levels)
        self.init_candidates = self.levels[0]
        for node in self.init_candidates:
            node.parent = None

        # show the max number of child node in the second-last level
        print(f'The max number of child node in the second-last level: {max([len(node.child) for node in self.levels[-2]])}')

        self.sparse_levels = []
        self.dense_levels = []
        self.location_levels = []
        self.radius_levels = []

    def prepare_tensors(self):
        sparse_level_num = 0
        dense_level_num = 0
        for level in tqdm(self.levels, desc='Preparing Bloom Filter Tensors'):
            num_bits = [len(node.bloom_filter) for node in level]
            max_bits = max(num_bits)
            avg_bits = sum(num_bits) / len(num_bits)
            loc = []
            rad = []
            if avg_bits / self.bloom_filter_dim > 0.05:
                dense_tensor = torch.zeros(len(level), self.bloom_filter_dim + 1, dtype=torch.float16)
                sparse_tensor = None
                dense_level_num += 1
            else:
                sparse_tensor = torch.empty(len(level), max_bits, dtype=torch.int16)
                dense_tensor = None
                sparse_level_num += 1
            for row_idx, node in enumerate(level):
                if avg_bits / self.bloom_filter_dim > 0.05:
                    # If the bloom filter is denser than 10%, we also build a dense tensor to store it
                    dense_tensor[row_idx, list(node.bloom_filter)] = 1
                else:
                    sorted_idx = sorted(list(node.bloom_filter))
                    sparse_tensor[row_idx, :len(sorted_idx)] = torch.tensor(sorted_idx, dtype=torch.int16)
                    sparse_tensor[row_idx, len(sorted_idx):] = -1
                loc.append(node.location)
                rad.append(node.radius)

            loc = torch.tensor(loc, dtype=torch.float32)
            rad = torch.tensor(rad, dtype=torch.float32)
            self.sparse_levels.append(sparse_tensor)
            self.location_levels.append(loc)
            self.radius_levels.append(rad)
            self.dense_levels.append(dense_tensor)
        print(f'Dense levels: {dense_level_num}, Sparse levels: {sparse_level_num}')

    def serialize(self, levels):
        with open(f'data_bin/{self.dataset_name}/tree.bin', 'wb') as f:
            f.write(struct.pack('I', len(levels)))
            for level in levels:
                f.write(struct.pack('I', len(level)))
                for cluster in level:
                    f.write(struct.pack('I', len(cluster)))
                    for node_id in cluster:
                        f.write(struct.pack('I', node_id))

    def deserialize(self):
        with open(f'data_bin/{self.dataset_name}/tree.bin', 'rb') as f:
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
    
    def build_from_clusters(self, clusters, prev_level):
        current_level = []
        for id_in_level, cluster in enumerate(clusters):
            new_node = TreeNode(set(), [], [], id_in_level=id_in_level)
            for node_id in cluster:
                new_node.add_child(prev_level[node_id])
            new_node.compute_radius()
            current_level.append(new_node)
        return current_level


    def get_truth_path(self, poi_idx):
        current_node = self.leaf_nodes[poi_idx]
        path = [current_node]
        while current_node.parent is not None:
            path.insert(0, current_node.parent)
            current_node = current_node.parent
        return path
    
    def load_candidates(self, file_path, num_rows, beam_widths):
        '''
            This function is used to deserialize the candidates from the C++ inference engine.
            The bin file: [query_id][depth][beam_width], unsigned int32
            candidates: [depth][query_id][beam_width]
        '''
        if not isinstance(beam_widths, list):
            beam_widths = [beam_widths] * (self.depth)
        candidates = [[None for i in range(num_rows)] for j in range(self.depth)]
        topk_list = [[] for _ in range(num_rows)]
        with open(file_path, 'rb') as f:
            for query_id in trange(num_rows, desc='Deserializing candidates'):
                for depth in range(self.depth):
                    beam_width = min(beam_widths[depth], len(self.levels[depth]))
                    node_list = struct.unpack(f'{beam_width}I', f.read(beam_width * 4))
                    candidates[depth][query_id] = node_list
                topk_list[query_id] = candidates[self.depth - 1][query_id]
        return candidates, topk_list

'''
    This script dedicates to the construction of the spatial tree for all datasets.
    NOTE: GeoGLUE is heavily anonymized (it has 2.8M POIs but only roughly 5,000 unique locations), 
        so we don't use k-means to construct the tree for GeoGLUE when parent node number is larger than 6000.

'''
import time
import math

from sklearn.cluster import MiniBatchKMeans
from collections import OrderedDict
from typing import List

class TreeNode:
    '''
        This tree node class only contains the location information.
        It is used to construct the tree. When actually running the model, 
        we use a different TreeNode class that contains the bloom filter and node radius.
    '''
    def __init__(self, location):
        self.location: list = location
        if len(location) == 2:
            self.min_x = location[0]
            self.max_x = location[0]
            self.min_y = location[1]
            self.max_y = location[1]
        else:
            self.max_x: float = -1e9
            self.min_x: float = 1e9
            self.max_y: float = -1e9
            self.min_y: float = 1e9
        self.child: List[TreeNode] = []
        self.parent: TreeNode = None

    def __repr__(self):
        return f'Node: {self.location}'
    
    def add_child(self, child):
        self.child.append(child)
        child.parent = self
        self.max_x = max(self.max_x, child.max_x)
        self.min_x = min(self.min_x, child.min_x)
        self.max_y = max(self.max_y, child.max_y)
        self.min_y = min(self.min_y, child.min_y)
    
    def compute_radius(self):
        self.radius = math.sqrt((self.max_x - self.min_x) ** 2 + (self.max_y - self.min_y) ** 2) / 2
        self.location = [(self.max_x + self.min_x) / 2, (self.max_y + self.min_y) / 2]

def build_kmeans_tree(dataset, poi_locs, width=8):
    leaf_nodes: List[TreeNode] = []
    num_pois = len(poi_locs)

    for i in range(num_pois):
        poi_node = TreeNode(poi_locs[i])
        leaf_nodes.append(poi_node)

    print('Constructing the tree via KMeans...')
    # if dataset is GeoGLUE, we do anti-anonymization
    if dataset == 'GeoGLUE':
        unique_locs = OrderedDict()
        for i, node in enumerate(leaf_nodes):
            loc = tuple(node.location)
            if loc not in unique_locs:
                unique_locs[loc] = []
            unique_locs[loc].append(i)

        # remove the locations with few nodes from the unique_locs
        unique_locs_few_nodes = OrderedDict()
        for loc, node_idxs in unique_locs.items():
            if len(node_idxs) < width // 2:
                unique_locs_few_nodes[loc] = node_idxs

        if len(unique_locs_few_nodes) > 0:

            for loc in unique_locs_few_nodes:
                del unique_locs[loc]

            coords = list(unique_locs.keys())
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            # if a location has no more than 8 nodes, we merge it with the closest location
            for loc, node_idxs in unique_locs_few_nodes.items():
                _, idx = tree.query(loc, k=1)
                closest_loc = coords[idx]
                unique_locs[closest_loc].extend(node_idxs)

        for loc, node_idxs in unique_locs.items():
            for node_idx in node_idxs:
                leaf_node = leaf_nodes[node_idx]
                leaf_node.max_x = loc[0]
                leaf_node.min_x = loc[0]
                leaf_node.max_y = loc[1]
                leaf_node.min_y = loc[1]
                leaf_node.location = loc

    levels = []
    cluster_dump = []
    prev_level = leaf_nodes
    while len(prev_level) > width * width:
        start = time.time()
        clusters = kmeans_cluster(dataset, prev_level, width)
        current_level = build_from_clusters(clusters, prev_level)
        levels.append(current_level)
        prev_level = current_level
        cluster_dump.append(clusters)
        end = time.time()
        print(f'Level with {len(current_level)} nodes constructed in {end - start:.3f}s')

    return cluster_dump

# Build the first level via merging the leaf nodes (sort by geohash)

def kmeans_cluster(dataset, nodes, width):

    unique_locs = OrderedDict()
    for i, node in enumerate(nodes):
        loc = tuple(node.location)
        if loc not in unique_locs:
            unique_locs[loc] = []
        unique_locs[loc].append(i)

    coords = list(unique_locs.keys())
    kmeans = MiniBatchKMeans(n_clusters=len(coords) // 6, random_state=0, batch_size=1000, n_init='auto')
    labels = kmeans.fit_predict(coords)
    initial_clusters = OrderedDict()
    for coord, label in zip(coords, labels):
        if label not in initial_clusters:
            initial_clusters[label] = []
        initial_clusters[label].extend(unique_locs[coord])
    # split all clusters with more than width nodes
    split_clusters = OrderedDict()
    for label, node_idxs in initial_clusters.items():
        if len(node_idxs) >= 10:
            # split node_idxs into multiple clusters
            num_splits = len(node_idxs) // 8
            for i in range(num_splits):
                split_clusters[len(split_clusters)] = node_idxs[i * width:(i + 1) * width]
            if len(node_idxs) % width > 0:
                split_clusters[len(split_clusters)] = node_idxs[num_splits * width:]
        else:
            split_clusters[len(split_clusters)] = node_idxs
    
    # There can be many clusters with less than width * 0.4 nodes, we merge them together
    standard_clusters = OrderedDict()
    small_clusters = OrderedDict()
    for i, cluster in enumerate(split_clusters.values()):
        if len(cluster) < 3:
            small_clusters[i] = cluster
        else:
            standard_clusters[i] = cluster
    
    # merge small clusters to the closest standard cluster
    standard_coords = []
    standard_labels = []
    for key in standard_clusters:
        first_node = nodes[standard_clusters[key][0]]
        standard_coords.append(first_node.location)
        standard_labels.append(key)

    small_coords = []
    small_labels = []       
    for key in small_clusters:
        first_node = nodes[small_clusters[key][0]]
        small_coords.append(first_node.location)
        small_labels.append(key)

    from scipy.spatial import cKDTree
    tree = cKDTree(standard_coords)
    _, idx = tree.query(small_coords, k=30)
    for i, nearest_standard_idxs in enumerate(idx):
        success = False
        for nearest_standard_idx in nearest_standard_idxs:
            standard_cluster = standard_clusters[standard_labels[nearest_standard_idx]]
            if len(standard_cluster) > width:
                continue
            else:
                success = True
                standard_clusters[standard_labels[nearest_standard_idx]].extend(small_clusters[small_labels[i]])
                break
        if not success:
            print(f'Warning: Failed to merge small cluster {small_labels[i]} with any standard cluster')

    result = list(standard_clusters.values())
    print(f'{len(result)} clusters after merging, with max size {max([len(cluster) for cluster in result])}')
    return result
    


def build_from_clusters(clusters, prev_level):
    current_level = []
    for cluster in clusters:
        new_node = TreeNode([])
        for node_id in cluster:
            new_node.add_child(prev_level[node_id])
        new_node.compute_radius()
        current_level.append(new_node)
    return current_level
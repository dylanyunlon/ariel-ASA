'''
    Developing logs:

    v1: The GeoBloom search engine should be a lightweight MLP that retrieves relevant geographic objects.
        - It accepts two bloom filters as input, one is from the query, the other is from the tree-node.
        - When the not trained, it should output the dot product of the two bloom filters exactly.
    
    v3: Works well on the Beijing dataset, but fails on GeoGLUE
    
    v6: Try to use Listwise loss (lambdarank)
        - Works well on Beijing
        - Interaction layer works as expected
        - Distance reweighter leads to overfitting (removed in v6 temporarily)
        
    v10 - Try the separated query rewriter
        - Separating the reranking and the retrieval process.
        Beijing NDCG@5: 0.605
        GeoGLUE beam_width = 2000: 0.468

    v12 - Remove the query rewriter as it negatively affect the result.
        - Special distance function design.
        
        I'm happy to announce that the model have achieved the best performance on all datasets.
        - However, the testing requires too much time. The reason is GPU-CPU bloom filter transfer and node collecting.
        - I noticed the Pikafish chess engine which may help.


    v14: Quantization & C++ inference engine.
        In this version, the model is trained in an "Eval -> Train -> Eval loop".
        In the future, the evaluation is expected to be done by the C++ inference engine.

        1. Try to use a unified model to do all the retrieval parts and a unique head to do the reranking part.
        2. Normalize the output of input encoding layer (avgpool) to unify the model at different depths.
        3. Use ClippedReLU instead of ReLU and limit the parameters in the optimizer.
           - Verified successfully on both Beijing and GeoGLUE.

        4. Implement the "Eval -> Train -> Eval loop" and test the performance on Beijing.
           - Verified successfully on Beijing.

        5. Ranking performance improvement
            - Truncate ranking loss to k=50 at the final depth
            - Add params d to the distance function
            - Separate the retrieval and reranking head and bottleneck.
            All effective.

    v15: Use KMeans to constuct the tree index; Use special tree design for GeoGLUE to bypass the anonymization.

        NOTE: So far, the best performance requires the following tuning:
        -   Consider node radius when computing query-node distance
        -   Listwise loss (lambdaLoss) for both retrieval and reranking
        -   Concatenate query and node embedding, then normalized by their bits
        -   Large input head and output head, LeakyReLU(x) + 1 activation
        -   TextSim = [a (TextScore - mean) / std + b].sigmoid(), a = 1, b = 0 before training.
            DistSim = -log(Distance + 1)
            Score = m * DistSim + n * TextSim - DistSim * TextSim, m = 1, n = 0 before training.
        -   Separate the retrieval and reranking layer. 
            Use full lambdaLoss for retrieval, truncated lambdaLoss (scheme=lambdaRank, k=50) for reranking.
        -   The average number of child should be as close as possible to the width of the tree.
            Use KMeans to build the Meituan tree index.
            Use the unique locations to build GeoGLUE's tree index.
            Use large inference beam width for GeoGLUE (the training width is fixed to 2000)
        -   When training the model, mix nodes from different depths within each mini-batch.

        v15 will be the last pure python version. We will implement the C++ inference engine in the next version.


    v16 - C++ inference engine.
        We now implement the C++ inference engine and test on all the datasets. The python part only includes the training part.

        Effective speed-up techniques:

        - Pre-embed the node bloom filters into 32-dim int32 vectors and store in the tree.
        - Embed the incomming query bloom filters into 32-dim int32 vectors. When searching, just do add and cliprelu.
        - Chunk the output of l3 into 8 parts and use 8 leakyrelu, so that AVX2 can be used for fast inference.
        - Pruning bloom filter common bits at the same depth.
        - Heap-sort based beam search.
        
        The python part consists of:
        1. POI / query serializer that writes the POI / query bloom filters and their locations into data_bin/{dataset}/poi.bin and {split}.bin file,
        2. Tree serializer that writes the tree index into data_bin/{dataset}/tree.bin file,
        3. GeoBloom writer functions that quantize the model parameters and write them into nnue/{dataset}_v{VERSION}.nnue file.
        4. Node encoder that encodes and quantizes the node bloom filters for the C++ inference engine (following the tree structure).

        The 1st and 2nd part replaces the previous pickle files. We store all files into data_bin folder, which is shared by the C++ inference engine.

        Besides,
        - Changed the torch bloom filters from torch.int16 to torch.int16 to save VRAMs.

        The remaining problems of quantization includes:
        1. The overflow of the encoder layer for queries (note that we have up to 1024 bits in one query).
            - Solved by int32 input weights. Note that each query will only pass through it once,
                so it shouldn' be an overhead (if it is, we reduce the hidden dim later).
            - As we will finally divide everything by the number of bits, this will have nearly no impact on the final result.
        2. The final reweight layer & LeakyReLU implementation's fast inference.
            - Solved by __mm256_maddubs_epi16 intrinsic function & >> 7 bit shift per intersection element.
        TODO: Add contextual information to the model (after C++ engine works)

        Results: 
        It works well on Beijing and Shanghai. 
            - Very fast (< 1ms each query); 
            - Much better than all methods.
        But it doesn't work well on GeoGLUE. 
            - Slow (30ms each query).
            - Only slightly better than best baseline; Much worse than Ziqi's method.

        We will first try to improve the model performance on GeoGLUE.

    v17 - Residual head.
        We add a residual head to the model to handle non-overlapping queries and further
        enhance the model's discriminative power to those similar queries (hard negatives).
        In this version, we use a shared head for both retrieval and reranking.
        The residual head will only add a negligible computation cost to the C++ inference engine.

        Minor changes:
        - Remove the relu in the final score computation as it does harm to the model performance.
        - Add a negative truncation to the recall loss to slightly improve the recall performance.

        The result shows that the residual head is consistently effective on all datasets.
            - Performance on GeoGLUE is still not satisfactory.
            - It may because there are too many false positives in the shallow layers 
            (We shouldn't punish them).
        
        We now have three options: 
        1. Continue to improve the model performance and speed on GeoGLUE.
            The mean challenge is to deal with the anonymization (Fake POIs).
            We can try to build a noise-robust model.
        2. Change a dataset.

    v18 - Context POIs
        We hereby add the context POIs to the model using the techniques found in context/geobloom_v18_rank_context_trial4.py.
        Some problems may occur in the recalling process, include:
            1. Initially, proper context POIs may not exist in the beam at the leaf layer. We may need to ensure their existence.
            2. Similarly, we may also need to ensure that the context's parent nodes are in the beam. This may be achieved by 
                1) adjusting the ground truth scores of the context to be the second largest score in the beam.
                2) imposing a separate loss on the context parent scores, and during the inference, expand the beam to include the context parent.

        In this first version, we just naively assume that the context and their parents are all in the beam, and will share all context scores and weights
        across all depths. We will try to improve this later if the performance is not as satisfactory as in v18_rank_context_trial4.

    v19 - Split all heads for different layers.
        It seems that the bottleneck is important. Simply splitting the heads for different layers doesn't work as good as v18.
        The result shows that the bottleneck is important. We should use different bottleneck for different layers.
        

        Experimental Results on Private Datasets, Xeon E5-2698 2.20GHz (* means the best result):

            Beijing: beam 200-300-300-300
            Total search time of all threads: 5.88955s, Query Per Second: 1606.41 (v17), 1247.86 (v18)
            =============== Intermediate Recall Scores ==============
            0.984915        0.939978        0.874999                    (GeoBloom v19 no context)*
            0.986297        0.935612        0.872766                    (GeoBloom v19)
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.825786        0.788883        0.643527        0.547722    (GeoBloom v19 no context)
            0.824922        0.786649        0.644604        0.549730    (GeoBloom v19)
            0.726300        0.663300        0.505600        0.414500    (Best Baseline TkQ-DPR_D)
            =========================================================

            Shanghai: beam 200-400-300-300
            Total search time of all threads: 12.1486s, Query Per Second: 1046.87(v17), 781.081(v18)
            =============== Intermediate Recall Scores ==============
            0.965070        0.932397        0.884622                    (GeoBloom v19 no context)*
            0.962212        0.928148        0.882186                    (GeoBloom v19)
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.826102        0.790903        0.658958        0.567857    (GeoBloom v19 no context)
            0.825824        0.794026        0.663791        0.572260    (GeoBloom v19)*
            0.771700        0.734800        0.574200        0.455600    (Best Baseline TkQ-DPR_D)
            =========================================================

        On public datasets: GeoGLUE: beam 6000-4000-4000-1000
            Total search time of all threads: 683.43s, Query Per Second: 29.2641(v17), 24.2718(v18)
            =============== Intermediate Recall Scores ==============
            0.908900        0.839600        0.786750                    (GeoBloom v19 no context)
            0.919650        0.861850        0.803650                    (GeoBloom v19)*
            ====================== Evaluation =======================
            Recall@20        Recall@10       NDCG@5          NDCG@1
            0.764550        0.736650        0.610570        0.513950    (GeoBloom v19 no context)
            0.792250        0.762950        0.634941        0.534300    (GeoBloom v19)*
            0.735700        0.701200        0.579700        0.484800    (Best Baseline TkQ-DPR_D)
            =========================================================

        The model structure and the performance is good enough for a new paper.
        We will now start the paper writing process and use a better CPU for single-thread speed testing.

        DATE: 2024-01-09

    v19_new - Faster Training Speed
        In this version, we use torch_sparse instead of vanilla torch.sparse_coo_tensor.
        This leads to a significant speed-up in the training process.
        Besides, we optimize the bloom filter for Beijing, Shanghai, and GeoGLUE-clean datasets.

'''
import math
import os
import time
import torch
import struct
import subprocess

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from torch.utils.cpp_extension import load

from bloom_filter_tree import BloomFilterTree
from lambdarank import lambdaLoss
from dataset import POIDataset, POIRetrievalDataset

# On Beijing and Shanghai, all user selections are equally important, there is no ranking priority.
# On GeoGLUE and GeoGLUE_clean, there is only one ground truth for each query.
# Hence, the following utility function can calculate NDCG faster than torch_metrics while with identical results.

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

VERSION = '19'

# 4*8192 = -32767 ~ 32768 can be exactly stored with torch.int16, so it saves VRAM
# And we can compare with other methods conveniently as they all uses fp16

# Compile and load the CUDA extension with Ninja
isin_cuda = load(
    name="isin_cuda",
    sources=["cuda/isin_cuda.cu"],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-lineinfo'],
    verbose=True
)

def isin_cuda_wrapper(elements, test_elements, padding_idx, dense=False):
    return isin_cuda.isin_cuda(elements, test_elements, padding_idx, dense)

class FakeReLU(nn.Module):
    def forward(self, x):
        # silu = F.silu(x)
        # relu = F.relu(x)
        # # Forward is relu, backward is silu
        # return silu + (relu - silu).detach()
        return F.relu(x + 1e-6)

class ZeroProjection(nn.Module):
    '''
        The nn.Embedding is too slow.
        We use this to replace it.
    '''
    def __init__(self, expanded_dim, hidden_dim):
        super(ZeroProjection, self).__init__()
        self.expanded_dim = expanded_dim
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.zeros(expanded_dim, hidden_dim, dtype=torch.float32))

    def forward(self, x):
        # assert torch.all(self.weight.data[-1,:] == 0)
        out = self.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

# Model definition
class GeoBloom(nn.Module):
    def __init__(self, bloom_filter_dim, depth=4, quantized_one=127.0, weight_scale=64.0, division_factor=16.0, d_threshold=1000.0):
        super(GeoBloom, self).__init__()
        self.depth = depth
        self.bloom_filter_dim = bloom_filter_dim
        self.expanded_dim = bloom_filter_dim + 1
        self.quantized_one = quantized_one
        self.weight_scale = weight_scale
        self.d_threshold = d_threshold

        # We use one common encoder for all depths and both queries and nodes
        # This encoder is especially large to ensure the model's capacity.

        # We follow the Stockfish NNUE initialization.

        sigma = math.sqrt(1/bloom_filter_dim)
        encoder_weight = torch.rand(self.expanded_dim, 256, dtype=torch.float32) * (2 * sigma) - sigma
        encoder_bias = torch.rand(256, dtype=torch.float32) * (2 * sigma) - sigma
        encoder_weight[-1, :] = 0

        # Setup the embedding bag for bloom filter embedding
        self.encoder = nn.EmbeddingBag(self.expanded_dim, 256, padding_idx=bloom_filter_dim, mode='sum', _weight=encoder_weight)
        self.encoder_bias = nn.Parameter(encoder_bias)

        self.bottleneck_list = [
            nn.Linear(512, 32) for _ in range(depth)
        ]
        self.bottleneck = nn.ModuleList(self.bottleneck_list)

        # We use experts to handle different depths when reweighting bloom filter's intersections.
        # Note: these decoders are serialized into int8 (i.e. 64x smaller than encoder), so they are tiny and won't largely affect the size.
        # But for fair comparison on disk usage with future baselines, we will write a separate function to do int16 quantization.
        # NOTE: The torch.compile has a bug that it can't handle the non-constant index for nn.ModuleList. 
        # Hence we use the python list, but with nn.ModuleList for model.parameters().
        self.rank_list = []
        for i in range(depth):
            self.rank_list.extend([
                nn.Linear(32, 32),
                ZeroProjection(self.expanded_dim, 32),
            ])
        self.rank = nn.ModuleList(self.rank_list)

        # Exactly the same structure as the decoder. It is possible to simplify the model by sharing some weights, but we leave this to future work.
        self.context_select_list = []
        for i in range(depth):
            self.context_select_list.extend([
                nn.Linear(32, 32),
                ZeroProjection(self.expanded_dim, 32),
            ])

        self.context_select = nn.ModuleList(self.context_select_list)

        self.context_rank_list = []
        for i in range(depth):
            self.context_rank_list.extend([
                nn.Linear(32, 32),
                ZeroProjection(self.expanded_dim, 32),
            ])
        self.context_rank = nn.ModuleList(self.context_rank_list)

        # The residual head. Similar to the context head, we use a separate head for each depth, but dim=1 for outputs.
        self.residual_list = []
        for i in range(depth):
            self.residual_list.extend([
                nn.Linear(32, 32),
                nn.Linear(32, 1, bias=False),
            ])
        self.residual = nn.ModuleList(self.residual_list)
        self.leaky_relu = nn.LeakyReLU(1.0 / division_factor)
        self.fake_relu = FakeReLU()

        # The division factor must be a power of 2.
        # So that we can use bit shift instead of division to ensure the speed on the C++ side.

        # Distance Modeling
        self.a = nn.Linear(depth, 1, bias=False)
        self.b = nn.Linear(depth, 1, bias=False)
        self.c = nn.Linear(depth, 1, bias=False)
        self.d = nn.Linear(depth, 1, bias=False)

        for i in range(depth):
            nn.init.zeros_(self.residual[2 * i + 1].weight)

        nn.init.ones_(self.a.weight)
        nn.init.zeros_(self.b.weight)
        nn.init.ones_(self.c.weight)
        nn.init.zeros_(self.d.weight)

        weight_bound = quantized_one / weight_scale
        self.weight_bound = weight_bound
        self.weight_clipping = [
            {'params' : [self.d.weight], 'min_weight' : 0, 'max_weight' : 127 },
        ]

        for i in range(depth):
            self.weight_clipping.append(
                {'params' : [self.bottleneck[i].weight], 'min_weight' : -weight_bound, 'max_weight' : weight_bound }
            )

        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(2 * depth):
                self.weight_clipping.append(
                    {'params' : [layer[i].weight], 'min_weight' : -weight_bound, 'max_weight' : weight_bound }
                )
    

    @torch.inference_mode()
    def serialize(self, path, size_test=False):
        '''
            This function is used to quantize and serialize the model parameters.
        '''

        if not size_test:
            # In production environment inference, we follows the NNUE quantization scheme, which yields slightly larger model file (~44 MB)
            serializer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int32).flatten().cpu().numpy().tobytes()
            serializer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int8).flatten().cpu().numpy().tobytes()
            serializer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int32).flatten().cpu().numpy().tobytes()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float32).tobytes()
        else:
            # In size test, we use all 16-bit format to compare with other baselines in fp16 (~40MB)
            # NOTE: This is only for theoretical comparison. the resulting model is not only larger but also can't correctly.
            serializer_encoder = lambda x: x.mul(self.quantized_one).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer8 = lambda x: x.mul(self.weight_scale).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer32 = lambda x: x.mul(self.quantized_one * self.weight_scale).round().to(torch.int16).flatten().cpu().numpy().tobytes()
            serializer_float = lambda x: x.flatten().cpu().numpy().astype(np.float16).tobytes()
            
        buf = bytearray()
        # header: tree depth.
        # we don't need to serialize the bloom filter length as it must match the query & poi Bloom filter dataset.
        buf.extend(struct.pack('H', self.depth))
        # encoder layer
        buf.extend(serializer_encoder(self.encoder.weight[:-1,:]))
        buf.extend(serializer_encoder(self.encoder_bias))

        # bottleneck
        for i in range(self.depth):
            buf.extend(serializer8(self.bottleneck[i].weight[:,:256]))

        # all heads
        for layer in [self.rank, self.context_select, self.context_rank]:
            for i in range(self.depth):
                buf.extend(serializer8(layer[2 * i].weight))
                buf.extend(serializer32(layer[2 * i].bias))
                buf.extend(serializer8(layer[2 * i + 1].weight[:-1,:]))

        layer = self.residual
        for i in range(self.depth):
            buf.extend(serializer8(layer[2 * i].weight))
            buf.extend(serializer32(layer[2 * i].bias))
            buf.extend(serializer8(layer[2 * i + 1].weight))

        # the parameter a, b, c, d
        buf.extend(serializer_float(self.a.weight))
        buf.extend(serializer_float(self.b.weight))
        buf.extend(serializer_float(self.c.weight))
        buf.extend(serializer_float(self.d.weight))

        # write the binary buffer into file
        with open(path, 'wb') as f:
            f.write(buf)

    def clip_weights(self):
        for group in self.weight_clipping:
            for p in group['params']:
                p_data_fp32 = p.data
                min_weight = group['min_weight']
                max_weight = group['max_weight']
                p_data_fp32.clamp_(min_weight, max_weight)
                p.data.copy_(p_data_fp32)

    def get_lr(self):
        lr_list = [
            {'params': self.encoder.parameters(), 'lr' : 1e-3},
            {'params': self.a.parameters(), 'lr': 1e-3},
            {'params': self.b.parameters(), 'lr': 1e-3},
            {'params': self.c.parameters(), 'lr': 1e-3},
            {'params': self.d.parameters(), 'lr': 1e-3},
        ]
        
        for i in range(self.depth - 1):
            lr_list.append({'params': self.bottleneck[i].parameters(), 'lr': 1e-3})
        lr_list.append({'params': self.bottleneck[-1].parameters(), 'lr': 0.002})

        for layer in [self.rank, self.context_select, self.context_rank, self.residual]:
            for i in range(2 * (self.depth - 1)):
                lr_list.append({'params': layer[i].parameters(), 'lr': 1e-3})
            # The last layer is the final reranking depth. We use a larger learning rate for it for faster convergence, 
            # as it is has relatively fewer training samples than retrieval stage.
            lr_list.append({'params': layer[-2].parameters(), 'lr': 0.002})
            lr_list.append({'params': layer[-1].parameters(), 'lr': 0.002})
        return lr_list
    
    @autocast()
    def forward(self, query_bloom_filter, node_sparse, node_dense, query_loc, node_loc, node_radius, depth):
        
        # query_bloom_filter: (query_bits, row, col), shape = (batch_size, self.bloom_filter_dim)
        # node_bloom_filter: [(node_bits, row, col), ...] (len = batch_size), shape = (batch_size, candidate_num, self.bloom_filter_dim)
        # node_dense: shape = (batch_size, candidate_num, self.bloom_filter_dim + 1)
        # query_loc: shape = (batch_size, 2)
        # node_loc: shape = (batch_size, candidate_num, 2)
        # node_radius: shape = (batch_size, candidate_num, 1)
        query_bits = (query_bloom_filter != self.bloom_filter_dim).sum(dim=-1).view(-1, 1)
        query_embedding = self.encoder(query_bloom_filter)
        query_embedding = (query_embedding + self.encoder_bias) / query_bits

        dense = node_dense is not None
        # NOTE: The information of node_bloom_filter all from the last layer of the tree.
        # So we don't compute the gradient for first several layers.
        if dense:
            B, K, _ = node_dense.shape
            node_bloom_filter = node_dense
            node_bits = node_dense.sum(dim=-1).view(B, K, 1)
            node_embedding = (node_dense @ self.encoder.weight + self.encoder_bias) / node_bits
        else:
            B, K, _ = node_sparse.shape
            node_bloom_filter = node_sparse
            node_bits = (node_sparse != self.bloom_filter_dim).sum(dim=-1).view(B, K, 1)
            node_embedding = self.encoder(node_sparse.view(B * K, -1))
            node_embedding = (node_embedding.view(B, K, -1) + self.encoder_bias) / node_bits

        query_embedding = torch.clamp(query_embedding, min=0, max=1)
        node_embedding = torch.clamp(node_embedding, min=0, max=1)

        hidden = torch.cat([query_embedding.unsqueeze(1).expand_as(node_embedding), node_embedding], dim=-1)
        hidden = self.bottleneck_list[depth](hidden)
        hidden = torch.clamp(hidden, min=0, max=1)
        H = hidden.shape[-1]


        intersection = query_bloom_filter.unsqueeze(1).expand(-1, K, -1)
        intersection_mask = isin_cuda_wrapper(intersection.reshape(B * K, -1), node_bloom_filter.view(B * K, -1), self.bloom_filter_dim, dense).view(B, K, -1)
        intersection = intersection.masked_fill(~intersection_mask, self.bloom_filter_dim)
        num_chunks = 8

        # Efficient ChunkedLeakyReLU
        hidden_rank = self.rank_list[2 * depth](hidden)
        hidden_rank = torch.clamp(hidden_rank, min=0, max=1) # shape = (B, K, 32)
        selected_columns = self.rank_list[2 * depth + 1](intersection) # shape = (B, K, L, 32)
        
        hidden_rank = hidden_rank.unsqueeze(-2).mul(selected_columns).view(B, K, -1, num_chunks, H // num_chunks)
        hidden_rank = self.leaky_relu(hidden_rank.sum(dim=-1)).sum(dim=-1) + 1
        hidden_rank = hidden_rank.masked_fill(~intersection_mask, 0)
        text_score = hidden_rank.sum(dim=-1)

        hidden_context_select = self.context_select_list[2 * depth](hidden)
        hidden_context_select = torch.clamp(hidden_context_select, min=0, max=1)
        selected_columns = self.context_select_list[2 * depth + 1](intersection) # shape = (B, K, L, 32)
        hidden_context_select = hidden_context_select.unsqueeze(-2).mul(selected_columns).view(B, K, -1, num_chunks, H // num_chunks)
        hidden_context_select = self.fake_relu(hidden_context_select.sum(dim=-1)).sum(dim=-1) + 1
        hidden_context_select = hidden_context_select.masked_fill(~intersection_mask, 0)
        context_select_score = hidden_context_select.sum(dim=-1)
        final_context_select_score = context_select_score
        del hidden_context_select
       
        # Context selection
        with torch.no_grad():
            node_dist = self.node_pairwise_distances(node_loc)
            mask = node_dist < self.d_threshold
            context_select_score = context_select_score.unsqueeze(1).expand_as(node_dist)
            context_select_score = context_select_score.masked_fill(~mask, -1e9)
            context_select_score = context_select_score.masked_fill(torch.eye(K, dtype=torch.bool, device=context_select_score.device).unsqueeze(0).expand(B, -1, -1), -1e9)
            best_context_idx = torch.argmax(context_select_score, dim=-1)
            # best_context_idx: [batch_size, num_nodes]
            context_bloom_filter = torch.gather(node_bloom_filter, dim=1, index=best_context_idx.unsqueeze(-1).expand_as(node_bloom_filter))
            # residual matching: (query - query * node) * context = query * (context - context * node) = query * (context - node).relu(). The last formular is faster.
            unmatched_bits = query_bloom_filter.unsqueeze(1).expand(-1, K, -1).masked_fill(intersection_mask, self.bloom_filter_dim)
            context_intersection_mask = isin_cuda_wrapper(unmatched_bits.view(B * K, -1), context_bloom_filter.view(B * K, -1), self.bloom_filter_dim, dense).view(B, K, -1)
            context_intersection = unmatched_bits.masked_fill(~context_intersection_mask, self.bloom_filter_dim)
        
        del query_bloom_filter, node_bloom_filter, context_bloom_filter

        hidden_context_rank = self.context_rank_list[2 * depth](hidden)
        hidden_context_rank = torch.clamp(hidden_context_rank, min=0, max=1)
        selected_columns = self.context_rank_list[2 * depth + 1](context_intersection) # shape = (B, K, L, 32)
        hidden_context_rank = hidden_context_rank.unsqueeze(-2).mul(selected_columns).view(B, K, -1, num_chunks, H // num_chunks)
        context_score = self.leaky_relu(hidden_context_rank.sum(dim=-1)).sum(dim=-1)
        context_score = context_score.masked_fill(~context_intersection_mask, 0).sum(dim=-1)
        del hidden_context_rank, context_intersection

        res_hidden = self.residual_list[2 * depth](hidden)
        res_hidden = torch.clamp(res_hidden, min=0, max=1)
        res_score = self.residual_list[2 * depth + 1](res_hidden)

        # We hereby add the residual head to handle the non-overlapping queries (i.e. intersection = 0)
        # Please be aware that this head ranges from [-64, 64] but intersection ranges from [0, query_bits].
        # We assume query bits is less than 1024, so we rescale the residual head to [-1024, 1024], i.e., multiply it by 16
        # NOTE: we can't rescale desc_sim by query bits; this will lead to instability.

        desc_sim = text_score + res_score.squeeze(-1) * 16 + context_score

        desc_std, desc_mean = torch.std_mean(desc_sim, dim=-1, keepdim=True)
        desc_sim = (desc_sim - desc_mean) / (desc_std + 1e-5)

        dist = torch.sum((query_loc.unsqueeze(1) - node_loc) ** 2, dim=-1).sqrt()
        dist = (dist - node_radius.squeeze(-1)).relu()

        dist_sim = - torch.log(dist + 1)
        final_score = (self.c.weight[0][depth] - (self.a.weight[0][depth] * desc_sim + self.b.weight[0][depth]).sigmoid()) * (dist_sim - self.d.weight[0][depth])

        return final_score, final_context_select_score
    
    
    @staticmethod
    def node_pairwise_distances(node_loc):
        # node_loc shape: [batch_size, num_nodes, 2]
        # Expand node_loc to shape [batch_size, num_nodes, 1, 2]
        node_loc_expanded = node_loc.unsqueeze(2)
        # Compute squared differences in both dimensions (x and y)
        diff_squared = (node_loc_expanded - node_loc_expanded.transpose(1, 2)) ** 2
        # Sum over the last dimension (x and y) and take the square root
        distances = torch.sqrt(diff_squared.sum(-1))

        return distances

    @autocast()
    @torch.inference_mode()
    def encode_node(self, sparse, dense, depth):
        if sparse is not None:
            node_bit_num = torch.sum(sparse != self.bloom_filter_dim, dim=-1).view(-1, 1)
            node_embedding = self.encoder(sparse)
            node_embedding = (node_embedding + self.encoder_bias) / node_bit_num
        else:
            node_embedding = dense @ self.encoder.weight + self.encoder_bias
            node_bit_num = dense.sum(dim=-1).view(-1, 1)
            node_embedding = node_embedding / node_bit_num
        node_embedding = torch.clamp(node_embedding, min=0, max=1)
        node_embedding = torch.matmul(node_embedding, self.bottleneck[depth].weight.T[256:])
        node_embedding = node_embedding + self.bottleneck[depth].bias
        return node_embedding
    
    def encode_tree(self, tree: BloomFilterTree):
        node_representations = []
        for depth in tqdm(range(tree.depth), desc='Encoding nodes'):
            sparse = tree.sparse_levels[depth]
            dense = tree.dense_levels[depth]
            # move it to cuda
            if sparse is not None:
                sparse = sparse.to(device='cuda', non_blocking=True).int()
                sparse[sparse == -1] = self.bloom_filter_dim
            if dense is not None:
                dense = dense.to(device='cuda', non_blocking=True)
            node_representations.append(self.encode_node(sparse, dense, depth).mul(127 * 64).round().to(dtype=torch.int32).cpu().numpy())
        node_representations = np.vstack(node_representations)
        torch.cuda.empty_cache()
        return node_representations

def train(model: GeoBloom, optimizer, tree: BloomFilterTree, train_beam_width, infer_beam_width, max_epochs, poi_dataset: POIDataset, batch_size, portion=None):

    bloom_filter_dim = model.bloom_filter_dim
    train_beam_width = [train_beam_width] * tree.depth if not isinstance(train_beam_width, list) else train_beam_width
    num_train_queries = len(poi_dataset.train_dataset)
    num_dev_queries = len(poi_dataset.dev_dataset)
    train_bloom_filter = poi_dataset.train_dataset.query_bloom_filters
    train_locs = poi_dataset.train_dataset.query_locs
    max_metrics = [0] * (tree.depth + 1)

    # We apply recall truncation on GeoGLUE as it is too noisy.
    retrieve_loss = lambda pred, truth: lambdaLoss(pred, truth, k=100, reduction='sum')
    rank_loss = lambda pred, truth: lambdaLoss(pred, truth, weighing_scheme='lambdaRank_scheme', k=30, reduction='sum')
    context_loss = lambda pred, truth: lambdaLoss(pred, truth, k=30, reduction='sum')
    
    ckpt_path = f'ckpt/{tree.dataset_name}_geobloom_v{VERSION}.pt' if portion is None else f'ckpt/{tree.dataset_name}_geobloom_v{VERSION}_{portion}.pt'
    transfer_path = f'model/tmp/{tree.dataset_name}_v{VERSION}/' if portion is None else f'model/tmp/{tree.dataset_name}_v{VERSION}_{portion}/'

    os.makedirs(transfer_path, exist_ok=True)
    os.makedirs('ckpt', exist_ok=True)

    train_beam_width_str = '-'.join([str(x) for x in train_beam_width] if isinstance(train_beam_width, list) else [
        str(train_beam_width) for _ in range(tree.depth)])
    infer_beam_width_str = '-'.join([str(x) for x in infer_beam_width] if isinstance(infer_beam_width, list) else [
        str(infer_beam_width) for _ in range(tree.depth)])
    
    # Prepare the truth nodes for evaluation
    train_truth_idx = [[set() for i in range(num_train_queries)] for j in range(tree.depth)]
    dev_truth_idx = [[set() for i in range(num_dev_queries)] for j in range(tree.depth)]

    for i in range(num_train_queries):
        for poi_idx in poi_dataset.train_dataset.truths[i]:
            path = tree.get_truth_path(poi_idx)
            for depth in range(tree.depth):
                train_truth_idx[depth][i].add(path[depth].id_in_level)
    for i in range(num_dev_queries):
        for poi_idx in poi_dataset.dev_dataset.truths[i]:
            path = tree.get_truth_path(poi_idx)
            for depth in range(tree.depth):
                dev_truth_idx[depth][i].add(path[depth].id_in_level)
    
    # We now record the time to reach the best dev ndcg.
    start_time = time.time()
    best_dev_time = 0

    for epoch in range(max_epochs):
        # We now use C++ inference engine to get the candidates.
        # First, we encode all the nodes into node.bin
        node_representations = model.encode_tree(tree).astype(np.int32)
        node_path = transfer_path + f'node_v{VERSION}.bin'
        with open(node_path, 'wb') as f:
            f.write(node_representations.tobytes())
        # Second, save the quantized model parameters
        model.serialize(transfer_path + f'nnue_v{VERSION}.bin')
        # Third, run the C++ inference engine to get the candidates
        # For faster inference we use multiple threads on the C++ side.
        threads = str(8)
        train_params = [f'nnue/v{VERSION}/nnue', tree.dataset_name, 'pytrain', threads, train_beam_width_str, transfer_path]
        if portion is not None:
            train_params.append(portion)
        subprocess.run(train_params)
        subprocess.run([f'nnue/v{VERSION}/nnue', tree.dataset_name, 'pydev', threads, infer_beam_width_str, transfer_path])

        # Fourth, deserialize the candidates
        next_train_candidates, train_topk = tree.load_candidates(transfer_path + f'train_nodes.bin', num_train_queries, train_beam_width)
        next_dev_candidates, dev_topk = tree.load_candidates(transfer_path + f'dev_nodes.bin', num_dev_queries, infer_beam_width)

        # Fifth, calculate train & dev recall and ndcg
        train_recalls, train_ndcgs = [], []
        dev_recalls, dev_ndcgs = [], []
        for depth in range(tree.depth):
            train_recalls.append([])
            for i in range(num_train_queries):
                hit = 0
                for node in next_train_candidates[depth][i]:
                    if node in train_truth_idx[depth][i]:
                        hit += 1
                    if hit >= len(train_truth_idx[depth][i]):
                        break
                train_recalls[-1].append(hit / len(train_truth_idx[depth][i]))
            train_recalls[-1] = np.mean(train_recalls[-1])

        for i in range(num_train_queries):
            pred = train_topk[i]
            truth = poi_dataset.train_dataset.truths[i]
            train_ndcgs.append(fast_ndcg(pred, truth, k=5))

        train_ndcgs = np.mean(train_ndcgs)

        for depth in range(tree.depth):
            dev_recalls.append([])
            for i in range(num_dev_queries):
                hit = 0
                for node in next_dev_candidates[depth][i]:
                    if node in dev_truth_idx[depth][i]:
                        hit += 1
                    if hit >= len(dev_truth_idx[depth][i]):
                        break
                dev_recalls[-1].append(hit / len(dev_truth_idx[depth][i]))
            dev_recalls[-1] = np.mean(dev_recalls[-1])

        for i in range(num_dev_queries):
            pred = dev_topk[i]
            truth = poi_dataset.dev_dataset.truths[i]
            dev_ndcgs.append(fast_ndcg(pred, truth, k=5))

        dev_ndcgs = np.mean(dev_ndcgs)

        print(f'==================== Epoch {epoch} ====================')
        print(f'Train recall: {train_recalls}, Train NDCG @ 5: {train_ndcgs}')
        print(f'Dev recall: {dev_recalls}, Dev NDCG @ 5: {dev_ndcgs}')
        print(f'Previous max metrics: {max_metrics}')
        # update the max metrics
        for depth in range(tree.depth):
            if dev_recalls[depth] > max_metrics[depth]:
                max_metrics[depth] = dev_recalls[depth]

        save = False
        if max_metrics[-1] < dev_ndcgs:
            max_metrics[-1] = dev_ndcgs
            save = True
            best_dev_time = time.time() - start_time
            print(f'New best dev time: {best_dev_time} (s)')
            # If trained, save the model
        
        print(f'Current max metrics: {max_metrics}')

        if save:
            torch.save(model.state_dict(), ckpt_path)
            print(f'Model saved to {ckpt_path}')

        model.train()
        # When training, we mix all the depth together to prevent catastrophic forgetting
        
        print('Preparing training data and targets...')

        scaler = GradScaler()   
        # We reconstruct the training dataloader for each depth.
        train_dataloader = []
        train_iter = []
        for depth in range(tree.depth):
            train_dataset = POIRetrievalDataset(train_bloom_filter, poi_dataset.num_slices, poi_dataset.num_bits, train_locs, tree.sparse_levels[depth], tree.dense_levels[depth], tree.location_levels[depth], tree.radius_levels[depth], train_truth_idx[depth], next_train_candidates[depth], train_beam_width[depth])
            train_dataloader.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=train_dataset.collate_fn))
            train_iter.append(iter(train_dataloader[-1]))

        # The dataloader has exactly the same number of batches for each depth.
        retrieve_losses = []
        rank_losses = []
        last_retrieve_loss = 0
        last_rank_loss = 0

        
        train_bar = tqdm(range(len(train_dataloader[0])), desc=f'Mixed training on {tree.depth} depths')
        for _ in train_bar:
            # reverse depth for debug: 
            for depth in range(tree.depth-1, -1, -1):
                try:
                    batch = next(train_iter[depth])
                except StopIteration:
                    train_iter[depth] = iter(train_dataloader[depth])
                    batch = next(train_iter[depth])
                query_bfs, query_loc, node_sparse, node_dense, node_locs, node_radius, target = batch
                query_bfs = query_bfs.to(device='cuda', non_blocking=True).int()
                query_bfs[query_bfs == -1] = bloom_filter_dim
                query_loc = query_loc.to(device='cuda', non_blocking=True)
                dense = node_dense is not None
                if node_sparse is not None:
                    node_sparse = node_sparse.to(device='cuda', non_blocking=True).int()
                    node_sparse[node_sparse == -1] = bloom_filter_dim
                    node_bfs = node_sparse
                if node_dense is not None:
                    node_dense = node_dense.to(device='cuda', non_blocking=True)
                    node_bfs = node_dense
                node_locs = node_locs.to(device='cuda', non_blocking=True)
                node_radius = node_radius.to(device='cuda', non_blocking=True)
                target = target.to(device='cuda', non_blocking=True)
                with torch.inference_mode():
                    B, K, _ = node_bfs.shape
                    # query_bfs shape = (B, L)
                    # select_mask shape = (B, K)
                    # node_bfs shape = (B, K, L)
                    top1_truth = torch.argmax(target, dim=-1) # shape = (B,)
                    truth_bfs = node_bfs[torch.arange(B), top1_truth] # shape = (B, L)
                    has_truth = (target != 0).any(dim=-1)
                    if not dense:
                        truth_bfs[~has_truth, :] = bloom_filter_dim
                    else:
                        truth_bfs[~has_truth, :] = 0
                    match_bits = isin_cuda_wrapper(query_bfs, truth_bfs, bloom_filter_dim, dense) # shape = (B, L)
                    missing_bits = query_bfs.masked_fill(match_bits, bloom_filter_dim)
                    # The context truth only contains those words that are not in the ground truth but in the query.
                    # We need to find the most similar node bloom filter in the slate, and select it as the context.
                    query_bits = (query_bfs != bloom_filter_dim).sum(dim=-1)
                    missing_bit_num = (missing_bits != bloom_filter_dim).sum(dim=-1)
                    easy_query = (missing_bit_num < query_bits * 0.5)
                    # NOTE: can't use repeat(K, 1) to replace unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1) as it gets wrong results.
                    context_rank_truth = isin_cuda_wrapper(missing_bits.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1), node_bfs.view(B * K, -1), bloom_filter_dim, dense)
                    context_rank_truth = context_rank_truth.view(B, K, -1).sum(dim=-1)
                    # NOTE: When there is no ground truth in beam, we also can't find a correct context.
                    context_rank_truth[~has_truth] = -1
                    context_rank_truth[easy_query] = -1
                    node_dist = GeoBloom.node_pairwise_distances(node_locs)
                    node_dist[target.unsqueeze(-2).expand_as(node_dist) == 0] = float('inf')
                    min_dist, _ = torch.min(node_dist, dim=-1)
                    dist_mask = min_dist > 1000
                    context_rank_truth[dist_mask] = -1
                    del truth_bfs, match_bits, missing_bits, node_dist, min_dist, dist_mask
                    
                score, context_score = model(query_bfs, node_sparse, node_dense, query_loc, node_locs, node_radius, depth)
                loss_fn = rank_loss if depth == tree.depth - 1 else retrieve_loss
                loss = loss_fn(score, target) + 0.1 * context_loss(context_score, context_rank_truth.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.clip_weights()
                loss_value = loss.item()
                if depth == tree.depth - 1:
                    last_rank_loss = loss_value
                    rank_losses.append(loss_value)
                else:
                    last_retrieve_loss = loss_value
                    retrieve_losses.append(loss_value)
                train_bar.set_postfix({'retrieve_loss': last_retrieve_loss, 'rank_loss': last_rank_loss})
        print(f'Epoch={epoch}, retrieve loss={np.mean(retrieve_losses)}, rank loss={np.mean(rank_losses)}')
        
    print(f'Max metrics: {max_metrics}')

    # We now quantize the best model and encode the node for further testing.
    model.load_state_dict(torch.load(ckpt_path))
    node_representations = model.encode_tree(tree).astype(np.int32)
    node_representations = np.vstack(node_representations).astype(np.int32)
    node_path = f'data_bin/{tree.dataset_name}/node_v{VERSION}.bin' 
    with open(node_path, 'wb') as f:
        f.write(node_representations.tobytes())
    print(f'Node representations saved to {node_path}')
    model.serialize(f'data_bin/{tree.dataset_name}/nnue_v{VERSION}.bin')
    print(f'Model serialized to data_bin/{tree.dataset_name}/nnue_v{VERSION}.bin')

    # run the C++ engine for testing
    threads = str(8)
    test_params = [f'nnue/v{VERSION}/nnue', tree.dataset_name, 'test', threads, infer_beam_width_str]
    subprocess.run(test_params)
    print(f'Best dev time: {best_dev_time} (s)')
    # append the best dev time to the result file if it exists
    result_path = f'result/{tree.dataset_name}_v{VERSION}_test.txt'
    if os.path.exists(result_path):
        with open(result_path, 'a') as f:
            f.write(f'Best dev time: {best_dev_time} (s)\n')
                
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--portion', type=str, default='1')
    parser.add_argument('--epochs', type=int, default=15)

    args = parser.parse_args()
    dataset = args.dataset
    task = args.task
    portion = args.portion
    portion = None if portion == '1' else portion
    max_epochs = args.epochs
    if dataset == 'GeoGLUE': # GeoGLUE has significant more noisy POIs, which requires longer Bloom filters.
        num_slices = 2
        num_bits = 16384
    else:
        num_slices = 2
        num_bits = 8192

    ckpt_path = f'ckpt/{dataset}_geobloom_v{VERSION}.pt' if portion is None else f'ckpt/{dataset}_geobloom_v{VERSION}_{portion}.pt'
    nnue_path = f'data_bin/{dataset}/nnue_v{VERSION}.bin'

    task_list = ['train', 'pytest', 'quantize', 'node', 'continue', 'size_test']

    assert task in task_list, f'The task should be in {task_list}'

    if task == 'quantize' or task == 'size_test':
        # load the model ready for quantization
        ckpt = torch.load(ckpt_path)
        # Identify the tensor - replace 'layer_name.weight' with the actual key
        if '_orig_mod.a.weight' in ckpt:
            depth = ckpt['_orig_mod.a.weight'].shape[1]
            model = GeoBloom(bloom_filter_dim=num_slices * num_bits, depth=depth).cuda()
            model = torch.compile(model)
        elif 'a.weight' in ckpt:
            depth = ckpt['a.weight'].shape[1]
            model = GeoBloom(bloom_filter_dim=num_slices * num_bits,depth=depth).cuda()
        else:
            print('Cannot find the depth of the model in the checkpoint.')
            exit()
        model.load_state_dict(ckpt)
        save_path = nnue_path if task != 'size_test' else f'data_bin/{dataset}/nnue_v{VERSION}_size_test.bin'
        model.serialize(save_path, size_test= task == 'size_test')
        print(f'Quantized model saved to {save_path}')
        if task == 'quantize':
            exit()

    # Create the data module and prepare the datasets
    batch_size = 8 if dataset == 'GeoGLUE' else 64
    poi_dataset = POIDataset(dataset, num_slices, num_bits, load_query=task!='node', portion=portion)

    train_beam_width = {
        'Beijing': 100,
        'Shanghai': 100,
        'GeoGLUE': 400,
        'GeoGLUE_clean': 400,
    }

    infer_beam_width = {
        'Beijing': 400,
        'Shanghai': 400,
        'GeoGLUE': 4000,
        'GeoGLUE_clean': 800,
    }

    # Initialize the model
    tree = BloomFilterTree(poi_dataset, width=8)
    tree.prepare_tensors()
    model = GeoBloom(bloom_filter_dim=num_slices * num_bits, depth=tree.depth).cuda()
    # model = torch.compile(model)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if task == 'train' or task == 'continue':
        if task == 'continue':
            model.load_state_dict(torch.load(ckpt_path))
        # Training loop
        # If your device doesn't support torch.compile, please comment the following line.
        train(model, 
            optimizer, 
            tree,
            train_beam_width[dataset], 
            infer_beam_width[dataset],
            max_epochs = max_epochs,
            poi_dataset = poi_dataset,
            batch_size = batch_size,
            portion = portion)

    elif task == 'node' or task == 'size_test':
        # load the current best model
        model.load_state_dict(torch.load(ckpt_path))
        node_representations = model.encode_tree(tree).astype(np.int32)

        if task == 'node':
            # In inference we simply use int32 precomputed embedding to following the NNUE scheme, reducing our workload.
            node_representations = np.vstack(node_representations).astype(np.int32)
            node_path = f'data_bin/{dataset}/node_v{VERSION}.bin' 
        else:
            # In the paper we compare with int16 embeddings. As the embedding dim=32, the size differences are small.
            # We will further adjust the C++ engine to support int16 embeddings.
            node_representations = np.vstack(node_representations).astype(np.int16)
            node_path = f'data_bin/{dataset}/node_v{VERSION}_size_test.bin'

        with open(node_path, 'wb') as f:
            f.write(node_representations.tobytes())
        print(f'Node representations saved to {node_path}')

        if task == 'size_test':
            # Measure and print the size of each file:
            bin_path = f'data_bin/{dataset}/'
            model_path = os.path.join(bin_path,f'nnue_v{VERSION}_size_test.bin')
            embedding_path = os.path.join(bin_path,f'node_v{VERSION}_size_test.bin')
            bloom_path = os.path.join(bin_path,f'poi.bin')
            tree_path = os.path.join(bin_path,f'tree.bin')
            # Helper function to convert bytes to megabytes with two decimal places
            def convert_size_to_mb(size_bytes):
                # Convert from bytes to megabytes and format the number with two decimals
                size_mb = size_bytes / (1024 * 1024)
                return f"{size_mb:.2f}"

            # Dictionary to store file sizes in MB
            file_sizes = {
                "Model": [],
                "Embeddings": [],
                "Bloom Filters": [],
                "Tree": []
            }

            # List of file paths for easier iteration
            file_paths = [model_path, embedding_path, bloom_path, tree_path]
            total_size = 0
            # Compute file sizes
            for path in file_paths:
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    total_size += size / (1024 * 1024)
                    if "nnue" in path:
                        file_sizes["Model"].append(convert_size_to_mb(size))
                    elif "node" in path:
                        file_sizes["Embeddings"].append(convert_size_to_mb(size))
                    elif "poi" in path:
                        file_sizes["Bloom Filters"].append(convert_size_to_mb(size))
                    elif "tree" in path:
                        file_sizes["Tree"].append(convert_size_to_mb(size))
                else:
                    print(f"File not found: {path}")

            # Printing the sizes in the specified format
            print('======== File Sizes in MB ========')
            print('Components')
            for key, sizes in file_sizes.items():
                print(f"{key:<15} {' '.join(sizes)}")
            print(f'Total - {total_size:.2f}')
            print('==================================')




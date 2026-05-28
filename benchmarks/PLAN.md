# Ariel-ASA Development Plan & NeurIPS Review Technical Narrative
## Claude Session Milestone Assignment (M001–M038)

---

## Part I: NeurIPS Review-Grade Technical Narrative

### §1 从 `SpTSpMKernelAOSImpl` 这个好例子开始

`upstream/blaeq/src/Kernel/SpTSpMKernel.cu:58` 定义了核心稀疏张量乘法 kernel：

```cuda
__global__ void SpTSpMKernelAOSImpl(
    const unsigned int next_mesh_nnz_nums, const size_t D,
    const double* d_P_vals, const double* d_curr_M_vals,
    const size_t* d_next_M_nnz_vals_P_row_ids,
    const size_t* d_next_M_nnz_vals_P_col_ids,
    double* d_next_M_vals)
```

该 kernel 使用 `UNROLL_FACTOR=4` 的 grid-stride loop，每个线程处理 4 个 AOS 格式的稀疏元素。这是 Ariel 空间查询的计算瓶颈所在。

### §2 遵循该模式实现新的 `BenchmarkCollector`，让 `QueryHandler.performQueryWithPreLoadPvals` 可以输出 per-query metrics，并能按 seed 分组写入 JSON

我们在 `benchmarks/collectors/spatial_bench.py` 中实现了 `SpatialBenchCollector`，其数据 schema 直接对齐 `data_demo/gradient_norm_24k_data.json` 的格式：

- X 轴：2000 个 step（query index）
- Y 轴：latency_ms / throughput / GPU memory
- 多 seed：3 seeds，键名 `seed_0`, `seed_1`, `seed_2`
- 多 method：AOS-Range, SOA-Range, AOS-KNN, SOA-KNN

对应 `QueryHandler::performQueryWithPreLoadPvals`（`upstream/blaeq/src/Query/Query.cuh:38`）的签名：
```cpp
QueryResult performQueryWithPreLoadPvals(
    const std::string& queryPath, QueryType qType,
    bool saveFineMesh = false,
    int maxQueryCount = std::numeric_limits<int>::max(),
    size_t K = 0);
```

### §3 引入 `NvtxProfiler`，使 nsys profile 能够对每个 multigrid 层级独立计时

`upstream/blaeq/src/utils/NVTXProfiler.cuh` 提供了 RAII 风格的 NVTX range 标注。我们在 `benchmarks/cuda/bench_timing.cuh` 中设计了 `BenchScope` 类，将 NVTX range 与 CUDA event 计时统一：

```cpp
BenchScope scope(acc, seed, query_idx, stream, "RangeQuery");
// kernel launches here — automatically timed
```

同时 `GridAsSparseMatrix::pre_allocate_d_vals`（`upstream/blaeq/src/Data_Structures/Data_Structures.cuh`）的预分配策略被用于 timer 对象的预分配。

### §4 整合 `isin_cuda_sparse_kernel` 的 batched lookup

`upstream/geobloom/cuda/isin_cuda.cu:37` 的 sparse kernel：
```cuda
__global__ void isin_cuda_sparse_kernel(
    const scalar_t* elements,
    const scalar_t* test_elements, ...)
```

使用 `binary_search`（`isin_cuda.cu:14`）在 GPU 上进行有序集合的批量查找。这与 BLAEQ 的空间索引查询形成互补：BLAEQ 处理几何空间的 multigrid 遍历，GeoBloom 处理文本-空间的 bloom filter 检索。整合后，`BloomFilterTree.prepare_tensors`（`upstream/geobloom/model/bloom_filter_tree.py:95`）支持 hierarchical spatial query 的 GPU-resident 索引。

### §5 增强 `CUDAKmeans::uploadDataToGPU` 的 bulk transfer pipeline

`upstream/blaeq/src/Kmeans/CUDAKmeans.cu:49`：
```cpp
void CUDAKmeans::uploadDataToGPU(
    const double* data, size_t N, size_t dim, bool is_aos)
```

这个函数负责将聚类数据从 host 传输到 device。benchmark 模块需要在传输前后采样 GPU 内存，使用 `cudaMemGetInfo` 记录。

### §6 完善 `SparseTensorConverter::Convert_Coo2Csc` 的 benchmark coverage

`upstream/blaeq/src/Data_Structures/Data_Structures.cuh` 中的格式转换器确保 COO→CSC 的正确性。Benchmark 需要覆盖两种 memory layout（AOS/SOA）× 两种 query type（Range/KNN）的全组合。

---

### 大厂 Infra 函数交叉引用（已在 /home/claude/infra-repos/ 验证）

| # | 项目 | 函数/类 | 文件路径 | Ariel 对应设计 |
|---|------|---------|----------|----------------|
| 1 | NCCL | `ncclAllReduce` | `nccl/src/collectives.cc:113` | 多 GPU 空间索引分片的 allreduce 同步 |
| 2 | CUTLASS | `GemmBlockwise` | `cutlass/include/cutlass/gemm/kernel/gemm_blockwise.h:134` | SpTSpM kernel 的 tile 分块策略 |
| 3 | FlashAttention | `flash_fwd_kernel` | `flash-attention/csrc/.../flash_fwd_launch_template.h:32` | fused attention 模式启发 fused spatial predicate |
| 4 | DeepSpeed | `DeepSpeedZeroOptimizer_Stage3` | `DeepSpeed/deepspeed/runtime/zero/stage3.py:136` | parameter shard 模式用于 index partition |
| 5 | Megatron | `forward` (TE) | `Megatron-LM/megatron/core/extensions/transformer_engine.py:645` | pipeline-parallel forward 启发 multigrid 层级调度 |
| 6 | vLLM | `PagedAttention` | `vllm/vllm/v1/attention/ops/paged_attn.py:15` | paged memory 管理用于 query batch 调度 |
| 7 | Triton | `JITFunction` | `triton/python/triton/runtime/jit.py:622` | JIT 编译自定义 spatial kernel |
| 8 | FlashInfer | `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` | `flashinfer/flashinfer/cascade.py:561` | cascaded spatial query 共享前缀优化 |
| 9 | ByteTransformer | `wmma_attention_long_kernel` | `ByteTransformer/.../attention_fused_long.cu:31` | WMMA tensor core 用于低精度空间计算 |
| 10 | Flux | `GemmRS_multinode` | `flux/python/flux/gemm_rs_sm80.py:86` | GEMM+ReduceScatter overlap 用于 multi-node indexing |
| 11 | LightSeq | `LSTransformerEncoderFunc` | `lightseq/.../transformer_encoder_layer.py:28` | fused encoder 模式启发 fused index builder |
| 12 | CCCL/CUB | `DeviceReduce` | `cccl/cub/cub/device/device_reduce.cuh:89` | device-level reduction 用于 spatial aggregation |
| 13 | Apex | `DistributedFusedAdam` | `apex/apex/contrib/optimizers/distributed_fused_adam.py:270` | 分布式优化器的 memory tracking 模式 |
| 14 | TransformerEngine | `fp8_gemm_enabled` | `TransformerEngine/.../debug/features/api.py:90` | FP8 量化路径用于低精度 spatial predicate |
| 15 | FasterTransformer | `applyTemperaturePenalty` | `FasterTransformer/.../sampling_penalty_kernels.cu:26` | per-element scaling kernel 模式 |
| 16 | JAX | `pjit` | `jax/jax/_src/pjit.py:671` | partitioned JIT 用于 TPU spatial query |
| 17 | PyTorch | `FullyShardedDataParallel` | `pytorch/torch/distributed/fsdp/fully_sharded_data_parallel.py:118` | FSDP shard 策略用于 index 分片 |
| 18 | MaxText | `train_step` | `maxtext/src/maxtext/trainers/pre_train/train.py:295` | TPU train step 的 benchmark 采集模式 |
| 19 | FairScale | FSDP/OSS | `fairscale/fairscale/` | 早期 FSDP 实现的 shard 策略参考 |
| 20 | effective_transformer | CUDA kernels | `effective_transformer/cuda/` | ByteDance 的 effective attention padding 策略 |

---

## Part II: 38 Claude Session 开发里程碑

### 第 1 位 Claude（当前）：M001–M003 ✅ Benchmark Schema & Collector 骨架

- **M001** — 分析 `data_demo/*.json` 的 X 轴维度 schema（steps, time_hours, per-seed curves）
- **M002** — 创建 `benchmarks/collectors/spatial_bench.py`（`SpatialBenchCollector`, `MultiMethodBenchmark`, `GPUMemoryProfiler`）
- **M003** — 创建 `benchmarks/cuda/bench_timing.cuh`（`CudaTimerPair`, `QueryTimingAccumulator`, `BenchScope`）

### 第 2 位 Claude：M004–M005 BLAEQ Integration

- **M004** — 修改 `upstream/blaeq/src/Query/Query.cuh`，在 `QueryHandler::performQueryWithPreLoadPvals` 中集成 `QueryTimingAccumulator`
- **M005** — 修改 `upstream/blaeq/CMakeLists.txt`，添加 benchmark target，链接 NVTX 和 CUDA events

### 第 3 位 Claude：M006–M007 SOA Kernel 变体

- **M006** — 实现 `SpTSpMKernelSOAImpl`（SOA 内存布局的 SpTSpM kernel），与 AOS 版本对照
- **M007** — 在 collector 中添加 AOS vs SOA 对比 panel

### 第 4 位 Claude：M008–M009 GeoBloom CUDA Benchmark

- **M008** — 为 `isin_cuda_sparse_kernel` 和 `isin_cuda_dense_kernel` 添加 CUDA event 计时
- **M009** — 创建 `benchmarks/collectors/geobloom_bench.py`，采集 bloom filter lookup 延迟

### 第 5 位 Claude：M010 KMeans GPU Benchmark

- **M010** — 为 `CUDAKmeans::run` 的每个迭代添加计时，输出 convergence curve 数据

### 第 6 位 Claude：M011 GPU Memory Profiling

- **M011** — 实现 `GPUMemoryProfiler` 的持续采样模式，支持 background thread 采集

### 第 7 位 Claude：M012 Multi-Resolution Grid Sweep

- **M012** — 实现 kx16/kx64/kx256/kx1024 四种分辨率的自动化 benchmark 矩阵

### 第 8 位 Claude：M013–M014 Data Format Validation

- **M013** — JSON schema validator：确保 benchmark 输出与 `data_demo/*.json` 格式完全兼容
- **M014** — 回归测试：将 `data_demo/gradient_norm_24k_data.json` 作为 golden reference

### 第 9 位 Claude：M015–M016 Plotting Pipeline

- **M015** — `benchmarks/analysis/plot_curves.py`：matplotlib 绘图，支持 mean ± std 带状图
- **M016** — 多 panel 对比图（类似 reversed_figure_data.json 的 kx16_iid vs kx256_iid 并排）

### 第 10 位 Claude：M017 NVTX Integration Testing

- **M017** — 验证 nsys profile 能捕获每个 multigrid 层级的 NVTX range

### 第 11 位 Claude：M018 COO→CSC 转换 Benchmark

- **M018** — 对 `SparseTensorConverter::Convert_Coo2Csc` 添加格式转换延迟采集

### 第 12 位 Claude：M019 Dataset Scalability Sweep

- **M019** — 自动化 N=1K/10K/100K/1M 数据集规模的 benchmark

### 第 13 位 Claude：M020 Dimension Scalability Sweep

- **M020** — 自动化 D=2/3/4/8/16 维度的 benchmark

### 第 14 位 Claude：M021 NCCL-style Multi-GPU Sharding

- **M021** — 参考 `ncclAllReduce` 设计 multi-GPU index partition 的通信 benchmark

### 第 15 位 Claude：M022 CUTLASS-style Tiled SpTSpM

- **M022** — 参考 `GemmBlockwise` 实现 tiled SpTSpM 变体

### 第 16 位 Claude：M023 FlashAttention-style Fused Kernel

- **M023** — 参考 `flash_fwd_kernel` 实现 fused spatial predicate kernel

### 第 17 位 Claude：M024 DeepSpeed ZeRO-style Index Partition

- **M024** — 参考 `DeepSpeedZeroOptimizer_Stage3` 实现索引分片 + 通信策略

### 第 18 位 Claude：M025 vLLM PagedAttention-style Query Scheduling

- **M025** — 参考 `PagedAttention` 实现 query batch 的 paged memory 管理

### 第 19 位 Claude：M026 Triton JIT Spatial Kernel

- **M026** — 参考 `JITFunction` 用 Triton 编写自定义 spatial query kernel

### 第 20 位 Claude：M027 FlashInfer Cascaded Query

- **M027** — 参考 `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` 实现 cascaded spatial query

### 第 21 位 Claude：M028 ByteTransformer WMMA Integration

- **M028** — 参考 `wmma_attention_long_kernel` 用 Tensor Core 加速低精度空间计算

### 第 22 位 Claude：M029 Flux Multi-Node Benchmark

- **M029** — 参考 `GemmRS_multinode` 实现 multi-node spatial indexing 的 comm-overlap benchmark

### 第 23 位 Claude：M030 LightSeq Fused Index Builder

- **M030** — 参考 `LSTransformerEncoderFunc` 实现 fused multigrid index construction

### 第 24 位 Claude：M031 CUB Device-Level Spatial Aggregation

- **M031** — 参考 `DeviceReduce` 实现 device-level spatial result aggregation

### 第 25 位 Claude：M032 Apex Memory Tracking

- **M032** — 参考 `DistributedFusedAdam` 的 memory tracking 完善 GPU 内存基线

### 第 26 位 Claude：M033 TransformerEngine FP8 Spatial Predicate

- **M033** — 参考 `fp8_gemm_enabled` 实现 FP8 量化空间判定

### 第 27 位 Claude：M034 FasterTransformer Scaling Kernel

- **M034** — 参考 `applyTemperaturePenalty` 的 per-element scaling 优化空间距离计算

### 第 28 位 Claude：M035 JAX pjit TPU Spatial Query

- **M035** — 参考 `pjit` 设计 TPU 上的分区空间查询

### 第 29 位 Claude：M036 PyTorch FSDP Index Sharding

- **M036** — 参考 `FullyShardedDataParallel` 实现 PyTorch 原生的索引分片

### 第 30 位 Claude：M037 End-to-End Integration

- **M037** — 将 M004–M036 的所有 benchmark 整合为统一的 CI pipeline

### 第 31 位 Claude：M038 Paper-Ready Data Generation

- **M038** — 在用户 GPU 服务器上运行完整 benchmark 矩阵，生成 NeurIPS 论文级数据

### 第 32–38 位 Claude（M039–M045）：Paper Writing & Ablation

- **M039** — Ablation study: kernel unroll factor sweep (1,2,4,8)
- **M040** — Ablation study: block size sweep (128,256,512,1024)
- **M041** — Ablation study: AOS vs SOA memory layout on different GPU architectures
- **M042** — Ablation study: FP64 vs FP32 vs FP16 precision impact
- **M043** — Comparison with baselines: R-tree, k-d tree, GIST
- **M044** — LaTeX figure generation from benchmark JSON
- **M045** — Final paper review and camera-ready preparation

---

## Part III: 文件清单与 Diff 计划

### 新增文件（M001–M003，第 1 位 Claude）

```
benchmarks/
├── __init__.py
├── collectors/
│   ├── __init__.py
│   └── spatial_bench.py          ← 核心 Python 采集器
├── cuda/
│   └── bench_timing.cuh          ← CUDA event timing 集成
├── configs/
│   └── (待 M012 添加)
├── analysis/
│   ├── __init__.py
│   └── (待 M015 添加)
└── PLAN.md                       ← 本文件
```

### 待修改文件（后续 Claude session）

| 文件 | Session | 修改内容 |
|------|---------|----------|
| `upstream/blaeq/src/Query/Query.cuh` | #2 | 添加 `QueryTimingAccumulator*` 成员 |
| `upstream/blaeq/src/Query/Query.cu` | #2 | 在 query loop 中集成 BenchScope |
| `upstream/blaeq/CMakeLists.txt` | #2 | 添加 benchmark target |
| `upstream/blaeq/src/Kernel/SpTSpMKernel.cu` | #3 | 添加 SOA kernel 变体 |
| `upstream/geobloom/cuda/isin_cuda.cu` | #4 | 添加 CUDA event 计时 |
| `upstream/blaeq/src/Kmeans/CUDAKmeans.cu` | #5 | per-iteration timing |

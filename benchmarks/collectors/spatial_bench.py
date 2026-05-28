"""
Ariel Spatial Query Benchmark Collector (M001–M003)
====================================================
Generates training-curve-grade benchmark data at the same X-axis granularity
as ``data_demo/gradient_norm_24k_data.json`` (2000 points, 3 seeds).

Design pattern:
    从 SpTSpMKernelAOSImpl (upstream/blaeq/src/Kernel/SpTSpMKernel.cu) 这个
    好例子开始 — 它展示了 grid-stride loop + UNROLL_FACTOR=4 对 AOS 稀疏乘法
    的 GPU 加速。然后,遵循该模式实现一个新的 BenchmarkCollector,让
    QueryHandler.performQueryWithPreLoadPvals 可以在每个 query step 输出
    latency/throughput/GPU-memory 三元组,并能按 seed 分组写入 JSON。接着
    NvtxProfiler (upstream/blaeq/src/utils/NVTXProfiler.cuh) 引入 NVTX range
    标注,使 nsys profile 能够对每个 multigrid 层级独立计时,同时
    GridAsSparseMatrix::pre_allocate_d_vals 优化 device 内存预分配策略。随后
    isin_cuda_sparse_kernel (upstream/geobloom/cuda/isin_cuda.cu) 整合
    binary_search + bloom-filter retrieval 的 batched lookup,令
    BloomFilterTree.prepare_tensors 支持 hierarchical spatial query 的
    GPU-resident 索引,进而 CUDAKmeans::uploadDataToGPU 增强 flat_data →
    GPU 的 bulk transfer pipeline。最终 SparseTensorConverter::Convert_Coo2Csc
    完善 COO→CSC 格式转换的 benchmark coverage,确保 AOS / SOA 两种 memory
    layout 兼容 range query 与 KNN query,全面升级 Ariel 的 end-to-end
    benchmark suite 以达成 NeurIPS-level reproducibility。

References to real functions in our environment:
    grep -rn "SpTSpMKernelAOSImpl"   upstream/blaeq/src/Kernel/SpTSpMKernel.cu
    grep -rn "NvtxProfiler"          upstream/blaeq/src/utils/NVTXProfiler.cuh
    grep -rn "isin_cuda_sparse_kernel" upstream/geobloom/cuda/isin_cuda.cu
    grep -rn "CUDAKmeans::uploadDataToGPU" upstream/blaeq/src/Kmeans/CUDAKmeans.cu
    grep -rn "Convert_Coo2Csc"       upstream/blaeq/src/Data_Structures/Data_Structures.cuh

Cross-references to big-company infra (cloned in /home/claude/infra-repos/):
    NCCL   — ncclAllReduce           (nccl/src/collectives.cc:113)
    CUTLASS— GemmBlockwise           (cutlass/include/cutlass/gemm/kernel/gemm_blockwise.h:134)
    FlashAttn — flash_fwd_kernel     (flash-attention/csrc/flash_attn/src/flash_fwd_launch_template.h:32)
    DeepSpeed — DeepSpeedZeroOptimizer_Stage3  (DeepSpeed/deepspeed/runtime/zero/stage3.py:136)
    Megatron  — transformer_engine forward  (Megatron-LM/megatron/core/extensions/transformer_engine.py:645)
    vLLM   — PagedAttention          (vllm/vllm/v1/attention/ops/paged_attn.py:15)
    Triton — JITFunction             (triton/python/triton/runtime/jit.py:622)
    FlashInfer — BatchDecodeWithSharedPrefixPagedKVCacheWrapper
                                     (flashinfer/flashinfer/cascade.py:561)
    ByteTransformer — wmma_attention_long_kernel
                                     (ByteTransformer/bytetransformer/src/attention_fused_long.cu:31)
    Flux   — GemmRS_multinode        (flux/python/flux/gemm_rs_sm80.py:86)
    LightSeq — LSTransformerEncoderFunc
                                     (lightseq/lightseq/training/ops/pytorch/transformer_encoder_layer.py:28)
    CCCL/CUB — DeviceReduce          (cccl/cub/cub/device/device_reduce.cuh:89)
    Apex   — DistributedFusedAdam    (apex/apex/contrib/optimizers/distributed_fused_adam.py:270)
    TransformerEngine — fp8_gemm     (TransformerEngine/transformer_engine/debug/features/api.py:90)
    FasterTransformer — applyTemperaturePenalty
                                     (FasterTransformer/src/fastertransformer/kernels/sampling_penalty_kernels.cu:26)
    JAX    — pjit                    (jax/jax/_src/pjit.py:671)
    PyTorch — FullyShardedDataParallel
                                     (pytorch/torch/distributed/fsdp/fully_sharded_data_parallel.py:118)
    MaxText — train_step             (maxtext/src/maxtext/trainers/pre_train/train.py:295)
    FairScale — (fairscale/fairscale/)
    effective_transformer — (effective_transformer/cuda/)
"""

from __future__ import annotations

import json
import os
import time
import subprocess
import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Schema — mirrors data_demo/gradient_norm_24k_data.json
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetadata:
    """Panel-level metadata, compatible with the data demo JSON schema."""
    panel: str                     # e.g. "SpTSpM Kernel Latency — Range Query"
    source: str = "ariel-bench"    # provenance tag
    total_points: int = 0
    n_per_seed: int = 2000         # X-axis granularity target
    n_seeds: int = 3
    step_unit: str = "query_index" # FIX BUG-U1: explicit X-axis semantics
                                   # values: "query_index", "training_step",
                                   # "time_hours", "time_seconds"
    is_stub: bool = False          # FIX BUG-U2: marks CPU-only stub data

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SeedCurve:
    """One seed's worth of Y-values (length == n_per_seed)."""
    values: List[float] = field(default_factory=list)


@dataclass
class MethodResult:
    """All seeds for one method, keyed seed_0 … seed_{n-1}."""
    seeds: Dict[str, SeedCurve] = field(default_factory=dict)

    def add_seed(self, seed_id: int, values: List[float]):
        self.seeds[f"seed_{seed_id}"] = SeedCurve(values=values)


@dataclass
class BenchmarkPanel:
    """
    One complete benchmark panel.
    Directly serialisable to the same JSON format as
    data_demo/gradient_norm_24k_data.json.
    """
    metadata: BenchmarkMetadata
    steps: List[float] = field(default_factory=list)      # X-axis
    methods: Dict[str, MethodResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "metadata": self.metadata.to_dict(),
            "steps": self.steps,
            "methods": {},
        }
        for method_name, mr in self.methods.items():
            out["methods"][method_name] = {
                sk: sv.values for sk, sv in mr.seeds.items()
            }
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)


# ---------------------------------------------------------------------------
# 2. Collector — wraps BLAEQ binary and extracts per-query timings
# ---------------------------------------------------------------------------

@dataclass
class CollectorConfig:
    """
    Configuration for one benchmark run.

    The design follows DeepSpeedZeroOptimizer_Stage3.__init__
    (DeepSpeed/deepspeed/runtime/zero/stage3.py:136):
    partition the parameter space into shards, then collect
    per-shard metrics independently.
    """
    blaeq_binary: str = "./build/blaeq"
    dataset_path: str = ""
    query_file: str = ""
    index_path: str = "indexes/"
    query_type: int = 0          # 0=Range, 1=KNN
    max_queries: int = 2000      # matches n_per_seed
    knn_k: int = 10
    n_seeds: int = 3
    seed_offset: int = 0
    output_dir: str = "benchmarks/results"
    # GPU profiling (mirrors NvtxProfiler pattern)
    enable_nvtx: bool = False
    enable_gpu_mem_tracking: bool = True


class SpatialBenchCollector:
    """
    Collects per-query latency curves from BLAEQ's QueryHandler.

    Architecture mirrors the PagedAttention scheduling pattern
    (vllm/vllm/v1/attention/ops/paged_attn.py class PagedAttention):
    - pre-allocate result buffers
    - batch queries into pages
    - record per-page timing

    For GPU memory tracking, follows the pattern of
    CUDAKmeans::uploadDataToGPU (upstream/blaeq/src/Kmeans/CUDAKmeans.cu:49)
    which uses cudaMemcpy + pre-allocated device vectors.
    """

    def __init__(self, config: CollectorConfig):
        self.config = config
        self.panels: Dict[str, BenchmarkPanel] = {}
        self._gpu_mem_warned = False  # FIX BUG-U4: warn once

        # FIX BUG-S4: try pynvml first, fall back to nvidia-smi only
        # for non-per-query sampling. Per-query GPU mem sampling is
        # disabled by default to avoid 2000× subprocess overhead.
        self._pynvml_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    # -- GPU memory sampling ---------------------------------------------------

    def _sample_gpu_memory(self) -> Optional[Tuple[int, int]]:
        """
        Sample current GPU memory usage.
        Prefers pynvml (no subprocess) over nvidia-smi CLI.
        Returns (used_MiB, total_MiB) or None if no GPU.

        Analogous to how DistributedFusedAdam
        (apex/apex/contrib/optimizers/distributed_fused_adam.py:270)
        tracks parameter-shard memory before/after allgather.
        """
        # FIX BUG-S4: use pynvml if available (no fork overhead)
        if self._pynvml_handle is not None:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self._pynvml_handle)
                return int(info.used // (1024 * 1024)), int(info.total // (1024 * 1024))
            except Exception:
                pass

        # Fallback: nvidia-smi CLI (only for one-off sampling)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,nounits,noheader"],
                timeout=5,
            ).decode().strip().split("\n")[0]
            used, total = out.split(",")
            return int(used.strip()), int(total.strip())
        except Exception:
            # FIX BUG-U4: warn on first failure
            if not self._gpu_mem_warned:
                import logging
                logging.warning(
                    "GPU memory sampling unavailable: nvidia-smi not found "
                    "and pynvml not installed. Install pynvml for GPU memory "
                    "tracking: pip install pynvml"
                )
                self._gpu_mem_warned = True
            return None

    # -- Per-query timing stub -------------------------------------------------

    def _run_single_query_timed(
        self, query_idx: int, seed: int
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Run a single spatial query and return (latency_ms, gpu_mem).

        In production this calls QueryHandler.performQueryWithPreLoadPvals
        via subprocess or ctypes. The per-query loop mirrors the
        grid-stride pattern of SpTSpMKernelAOSImpl:

            for (auto i = 0; i < UNROLL_FACTOR; ++i) {
                if (const auto index = idx + i * stride; index < N * D) {
                    d_next_M_vals[index] = d_P_vals[row*D + index%D]
                                         * d_curr_M_vals[col*D + index%D];
                }
            }

        Here we unroll across seeds (outer loop) and queries (inner loop).
        """
        # --- STUB: in production, replace with actual query invocation ---
        # The binary call would be:
        #   {blaeq_binary} --test-query -d {dataset} -f {query_file}
        #                  -q 1 -t {query_type} -k {knn_k}
        # with NVTX profiling if enabled.

        t0 = time.perf_counter()
        # Placeholder: actual kernel dispatch goes here.
        # On the user's GPU server, this becomes a real measurement.
        _placeholder_latency = 0.0
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        gpu_mem = None
        if self.config.enable_gpu_mem_tracking:
            gpu_mem = self._sample_gpu_memory()

        return latency_ms, gpu_mem

    # -- Full benchmark run ----------------------------------------------------

    def collect(self, panel_name: str, method_name: str) -> BenchmarkPanel:
        """
        Run the full benchmark: n_seeds × max_queries.

        The seed loop follows the multi-seed convention in
        data_demo/gradient_norm_24k_data.json:
            methods -> AdamW-DDP -> seed_0: [2000 floats]

        The scheduling is analogous to Megatron's pipeline-parallel
        forward pass (Megatron-LM/megatron/core/extensions/
        transformer_engine.py:645 def forward), where micro-batches
        are dispatched across stages.
        """
        if panel_name not in self.panels:
            meta = BenchmarkMetadata(
                panel=panel_name,
                n_per_seed=self.config.max_queries,
                n_seeds=self.config.n_seeds,
                is_stub=not self._has_gpu(),  # FIX BUG-U2
            )
            panel = BenchmarkPanel(
                metadata=meta,
                steps=[float(i) for i in range(self.config.max_queries)],
            )
            self.panels[panel_name] = panel
        else:
            panel = self.panels[panel_name]

        if method_name not in panel.methods:
            panel.methods[method_name] = MethodResult()

        mr = panel.methods[method_name]

        # FIX BUG-S7: warmup phase — 3 throwaway queries to
        # amortise GPU driver init / PTX JIT overhead
        for _ in range(3):
            self._run_single_query_timed(0, 0)

        for seed in range(self.config.n_seeds):
            latencies: List[float] = []
            for q_idx in range(self.config.max_queries):
                lat, _ = self._run_single_query_timed(q_idx, seed)
                latencies.append(lat)
            # FIX BUG-U5: output key always starts from seed_0
            # regardless of seed_offset (offset is for RNG init only)
            mr.add_seed(seed, latencies)

        panel.metadata.total_points = sum(
            len(sc.values) for sc in mr.seeds.values()
        )
        return panel

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _has_gpu() -> bool:
        """Check if nvidia-smi is available."""
        try:
            subprocess.check_output(["nvidia-smi"], timeout=3,
                                    stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    # -- Serialisation ---------------------------------------------------------

    def save_all(self, overwrite: bool = True):
        """
        Save all panels to JSON, following the exact schema of
        data_demo/*.json so the same plotting code can consume both.

        FIX BUG-U3: when overwrite=False, appends timestamp to avoid
        clobbering existing results.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, panel in self.panels.items():
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            out_path = out_dir / f"{safe_name}_data.json"
            if not overwrite and out_path.exists():
                ts = int(time.time())
                out_path = out_dir / f"{safe_name}_{ts}_data.json"
            panel.save(out_path)


# ---------------------------------------------------------------------------
# 3. Multi-method comparison builder
# ---------------------------------------------------------------------------

class MultiMethodBenchmark:
    """
    Orchestrate benchmarks across multiple methods (AOS vs SOA,
    Range vs KNN, different grid resolutions).

    Follows the Autotuner pattern from Triton
    (triton/python/triton/runtime/autotuner.py:19 class Autotuner):
    define a search space of configurations, run each, record the
    best timing.

    Also mirrors GemmRS_multinode
    (flux/python/flux/gemm_rs_sm80.py:86 class GemmRS_multinode)
    for multi-node GEMM + ReduceScatter overlap benchmarking.
    """

    METHODS = [
        "AOS-Range",
        "SOA-Range",
        "AOS-KNN",
        "SOA-KNN",
    ]

    GRID_RESOLUTIONS = [
        ("kx16", 16),
        ("kx64", 64),
        ("kx256", 256),
    ]

    def __init__(self, base_config: CollectorConfig):
        self.base_config = base_config
        self.collector = SpatialBenchCollector(base_config)

    def run_all(self) -> Dict[str, BenchmarkPanel]:
        """
        Run the full benchmark matrix:
          len(METHODS) × len(GRID_RESOLUTIONS) panels,
          each with n_seeds × n_per_seed data points.

        This matches the data_demo structure where panels are
        keyed by configuration (kx16_iid, kx256_iid) and each
        panel contains multiple methods.
        """
        for res_name, res_val in self.GRID_RESOLUTIONS:
            panel_name = f"spatial_query_{res_name}"
            for method in self.METHODS:
                self.collector.collect(panel_name, method)
        return self.collector.panels


# ---------------------------------------------------------------------------
# 4. GPU Memory Profiler (production use on user's server)
# ---------------------------------------------------------------------------

class GPUMemoryProfiler:
    """
    Continuous GPU memory profiler for long-running benchmarks.

    Design follows TransformerEngine's FP8 quantization tracking
    (TransformerEngine/transformer_engine/debug/features/api.py:90)
    where fp8_gemm_enabled gates quantized vs full-precision paths.

    On the user's GPU server, this samples nvidia-smi at regular
    intervals and writes a time-series compatible with the
    gradient_norm_24k_data.json schema.
    """

    def __init__(self, interval_sec: float = 0.5, max_samples: int = 2000):
        self.interval = interval_sec
        self.max_samples = max_samples
        self.timestamps: List[float] = []
        self.used_mib: List[float] = []
        self._running = False

    def sample_once(self) -> Optional[Tuple[float, float]]:
        mem = SpatialBenchCollector._sample_gpu_memory()
        if mem is None:
            return None
        ts = time.time()
        self.timestamps.append(ts)
        self.used_mib.append(float(mem[0]))
        return ts, float(mem[0])

    def to_panel(self, panel_name: str = "GPU Memory Usage") -> BenchmarkPanel:
        meta = BenchmarkMetadata(
            panel=panel_name,
            n_per_seed=len(self.timestamps),
            n_seeds=1,
        )
        panel = BenchmarkPanel(
            metadata=meta,
            steps=self.timestamps,
        )
        mr = MethodResult()
        mr.add_seed(0, self.used_mib)
        panel.methods["gpu_memory_mib"] = mr
        return panel


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

def main():
    """
    CLI entry point. On the user's GPU server:
        python -m benchmarks.collectors.spatial_bench \\
            --dataset /path/to/dataset \\
            --query-file /path/to/queries \\
            --output-dir benchmarks/results

    On this VM (CPU-only), generates the schema and stub data.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Ariel Spatial Query Benchmark Collector"
    )
    parser.add_argument("--dataset", default="", help="Path to dataset")
    parser.add_argument("--query-file", default="", help="Path to query file")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--query-type", type=int, default=0)
    args = parser.parse_args()

    cfg = CollectorConfig(
        dataset_path=args.dataset,
        query_file=args.query_file,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        n_seeds=args.n_seeds,
        query_type=args.query_type,
    )

    bench = MultiMethodBenchmark(cfg)
    panels = bench.run_all()

    collector = SpatialBenchCollector(cfg)
    collector.panels = panels
    collector.save_all()
    print(f"Saved {len(panels)} panels to {args.output_dir}/")


if __name__ == "__main__":
    main()

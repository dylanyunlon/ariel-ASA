/**
 * benchmarks/cuda/bench_timing.cuh — Per-query CUDA event timing
 * ================================================================
 *
 * Integrates with:
 *   - NvtxProfiler       (upstream/blaeq/src/utils/NVTXProfiler.cuh)
 *   - SpTSpMGPUKernelAOS  (upstream/blaeq/src/Kernel/SpTSpMKernel.cu:40)
 *   - QueryHandler        (upstream/blaeq/src/Query/Query.cuh:23)
 *
 * Design follows CUB's DeviceReduce
 * (cccl/cub/cub/device/device_reduce.cuh:89 struct DeviceReduce)
 * pattern: static dispatch → kernel launch → synchronize → read back.
 *
 * Also references CUTLASS GemmBlockwise
 * (cutlass/include/cutlass/gemm/kernel/gemm_blockwise.h:187)
 * for the idea of tiling benchmark iterations into blocks.
 */

#ifndef ARIEL_BENCH_TIMING_CUH
#define ARIEL_BENCH_TIMING_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <chrono>

// FIX BUG-S5: conditional NVTX support
#ifdef ARIEL_USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace ariel_bench {

// ---------------------------------------------------------------------------
// CUDA event pair for bracketing a kernel or host-device transfer
// ---------------------------------------------------------------------------
struct CudaTimerPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float elapsed_ms = 0.0f;
    bool recorded = false;

    CudaTimerPair() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    // FIX BUG-S1: check nullptr before destroy (moved-from objects)
    ~CudaTimerPair() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }

    // Non-copyable, movable
    CudaTimerPair(const CudaTimerPair&) = delete;
    CudaTimerPair& operator=(const CudaTimerPair&) = delete;
    CudaTimerPair(CudaTimerPair&& o) noexcept
        : start(o.start), stop(o.stop), elapsed_ms(o.elapsed_ms),
          recorded(o.recorded) {
        o.start = nullptr;
        o.stop = nullptr;
        o.recorded = false;
    }

    // FIX BUG-S2: move assignment for vector reallocation safety
    CudaTimerPair& operator=(CudaTimerPair&& o) noexcept {
        if (this != &o) {
            if (start) cudaEventDestroy(start);
            if (stop) cudaEventDestroy(stop);
            start = o.start;
            stop = o.stop;
            elapsed_ms = o.elapsed_ms;
            recorded = o.recorded;
            o.start = nullptr;
            o.stop = nullptr;
            o.recorded = false;
        }
        return *this;
    }

    void record_start(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    void record_stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
        recorded = true;
    }

    float synchronize_and_get_ms() {
        if (!recorded) return 0.0f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms;
    }
};

// ---------------------------------------------------------------------------
// Per-query timing accumulator
// ---------------------------------------------------------------------------

/**
 * QueryTimingAccumulator — records per-query latencies across seeds.
 *
 * Schema matches data_demo/gradient_norm_24k_data.json:
 *   steps:   [0.0, 1.0, ..., N-1]
 *   methods: { "method_name": { "seed_0": [latency_ms, ...], ... } }
 *
 * Usage in QueryHandler::performQueryWithPreLoadPvals:
 *
 *   QueryTimingAccumulator acc("SpTSpM-AOS", n_queries, n_seeds);
 *   for (int seed = 0; seed < n_seeds; seed++) {
 *       for (int q = 0; q < n_queries; q++) {
 *           auto& timer = acc.get_timer(seed, q);
 *           timer.record_start(stream);
 *           SpTSpMGPUKernelAOS(...);   // the actual kernel
 *           timer.record_stop(stream);
 *       }
 *   }
 *   acc.synchronize_all();
 *   acc.save_json("benchmarks/results/sptm_aos_data.json");
 */
class QueryTimingAccumulator {
public:
    QueryTimingAccumulator(
        const std::string& method_name,
        size_t n_queries,
        size_t n_seeds
    )
        : method_name_(method_name),
          n_queries_(n_queries),
          n_seeds_(n_seeds)
    {
        // Pre-allocate timer pairs: seeds × queries
        // Mirrors CUDAKmeans::uploadDataToGPU pre-allocation strategy
        timers_.resize(n_seeds);
        for (auto& seed_timers : timers_) {
            seed_timers.reserve(n_queries);
            for (size_t i = 0; i < n_queries; i++) {
                seed_timers.emplace_back();
            }
        }
    }

    CudaTimerPair& get_timer(size_t seed, size_t query_idx) {
        assert(seed < n_seeds_ && query_idx < n_queries_);
        return timers_[seed][query_idx];
    }

    // Accessors (added M004) so callers can bounds-check query_idx/seed before
    // calling get_timer(), whose assert would otherwise fire in debug builds.
    size_t n_queries() const { return n_queries_; }
    size_t n_seeds() const { return n_seeds_; }

    void synchronize_all() {
        for (auto& seed_timers : timers_) {
            for (auto& t : seed_timers) {
                t.synchronize_and_get_ms();
            }
        }
    }

    /**
     * Save to JSON matching data_demo schema.
     *
     * Output format:
     * {
     *   "metadata": { "panel": "...", "n_per_seed": N, "n_seeds": S },
     *   "steps": [0.0, 1.0, ...],
     *   "methods": {
     *     "SpTSpM-AOS": {
     *       "seed_0": [0.123, 0.456, ...],
     *       "seed_1": [0.789, ...],
     *       ...
     *     }
     *   }
     * }
     */
    void save_json(const std::string& filepath) const {
        std::ofstream ofs(filepath);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open " << filepath << std::endl;
            return;
        }

        // FIX BUG-S3: consistent float precision
        ofs << std::fixed << std::setprecision(6);

        ofs << "{\n";

        // metadata
        ofs << "  \"metadata\": {\n"
            << "    \"panel\": \"" << method_name_ << " Latency\",\n"
            << "    \"source\": \"ariel-bench-cuda\",\n"
            << "    \"total_points\": " << (n_seeds_ * n_queries_) << ",\n"
            << "    \"n_per_seed\": " << n_queries_ << ",\n"
            << "    \"n_seeds\": " << n_seeds_ << "\n"
            << "  },\n";

        // steps
        ofs << "  \"steps\": [";
        for (size_t i = 0; i < n_queries_; i++) {
            if (i > 0) ofs << ", ";
            ofs << static_cast<double>(i);
        }
        ofs << "],\n";

        // methods
        ofs << "  \"methods\": {\n"
            << "    \"" << method_name_ << "\": {\n";
        for (size_t s = 0; s < n_seeds_; s++) {
            ofs << "      \"seed_" << s << "\": [";
            for (size_t q = 0; q < n_queries_; q++) {
                if (q > 0) ofs << ", ";
                ofs << timers_[s][q].elapsed_ms;
            }
            ofs << "]";
            if (s + 1 < n_seeds_) ofs << ",";
            ofs << "\n";
        }
        ofs << "    }\n  }\n}\n";
        ofs.close();
    }

    // -- GPU memory snapshot --------------------------------------------------

    struct GpuMemSnapshot {
        size_t free_bytes;
        size_t total_bytes;
    };

    static GpuMemSnapshot snapshot_gpu_memory() {
        GpuMemSnapshot snap{};
        cudaMemGetInfo(&snap.free_bytes, &snap.total_bytes);
        return snap;
    }

private:
    std::string method_name_;
    size_t n_queries_;
    size_t n_seeds_;
    // timers_[seed][query_idx]
    std::vector<std::vector<CudaTimerPair>> timers_;
};

// ---------------------------------------------------------------------------
// RAII benchmark scope — combines NVTX + CUDA event timing
// ---------------------------------------------------------------------------

/**
 * BenchScope — RAII guard that:
 *   1. Pushes an NVTX range (for nsys/ncu profiling)
 *   2. Records CUDA start/stop events
 *   3. Writes latency to an accumulator
 *
 * Usage (always stack-allocate; never heap + delete — that defeats RAII):
 *   {
 *       BenchScope scope(acc, seed, query_idx, stream, "RangeQuery");
 *       // ... kernel launches ...
 *   }  // stop event recorded, NVTX range popped
 *
 * Disabled form (M004 review fix BUG-4): construct with a nullptr accumulator
 * to get a no-op scope. This lets callers stack-allocate unconditionally
 * instead of `new BenchScope(...)` + `delete` (which leaks the CUDA events if
 * an exception is thrown between new and delete):
 *   {
 *       BenchScope scope(maybe_null_acc, seed, q, stream, name); // no-op if null
 *       ...
 *   }
 */
class BenchScope {
public:
    // Enabled form: bind to a live accumulator's timer.
    BenchScope(
        QueryTimingAccumulator& acc,
        size_t seed,
        size_t query_idx,
        cudaStream_t stream = 0,
        [[maybe_unused]] const char* nvtx_name = nullptr
    )
        : timer_(&acc.get_timer(seed, query_idx)),
          stream_(stream)
#ifdef ARIEL_USE_NVTX
        , nvtx_active_(false)
#endif
    {
#ifdef ARIEL_USE_NVTX
        if (nvtx_name) {
            nvtxRangePushA(nvtx_name);
            nvtx_active_ = true;
        }
#endif
        timer_->record_start(stream_);
    }

    // Disabled form (BUG-4 fix): nullptr accumulator -> no-op scope. Stack-safe.
    BenchScope(
        QueryTimingAccumulator* acc,
        size_t seed,
        size_t query_idx,
        cudaStream_t stream = 0,
        [[maybe_unused]] const char* nvtx_name = nullptr
    )
        : timer_(nullptr),
          stream_(stream)
#ifdef ARIEL_USE_NVTX
        , nvtx_active_(false)
#endif
    {
        if (acc != nullptr) {
            timer_ = &acc->get_timer(seed, query_idx);
#ifdef ARIEL_USE_NVTX
            if (nvtx_name) {
                nvtxRangePushA(nvtx_name);
                nvtx_active_ = true;
            }
#endif
            timer_->record_start(stream_);
        }
    }

    ~BenchScope() {
        if (timer_ != nullptr) {
            timer_->record_stop(stream_);
        }
#ifdef ARIEL_USE_NVTX
        if (nvtx_active_) {
            nvtxRangePop();
        }
#endif
    }

    BenchScope(const BenchScope&) = delete;
    BenchScope& operator=(const BenchScope&) = delete;

private:
    CudaTimerPair* timer_;   // nullptr => disabled no-op scope
    cudaStream_t stream_;
#ifdef ARIEL_USE_NVTX
    bool nvtx_active_;
#endif
};

}  // namespace ariel_bench

#endif  // ARIEL_BENCH_TIMING_CUH

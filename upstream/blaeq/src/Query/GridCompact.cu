//
// Created by Mgepahmge on 2025/12/17.
//

#include "GridCompact.cuh"

#include <filesystem>
#include <thrust/iterator/detail/device_system_tag.h>

#include "compact.cuh"
#include "src/Data_Structures/Data_Structures.cuh"

#include "src/utils/NVTXProfiler.cuh"



template <int Iter>
__global__ void gridCompactPrefixKernel(double* outputData, size_t* outputIndex, const double* inputData, const size_t* inputIndex, const bool* mask, const unsigned int* processorCounts,
                                    const unsigned int dataSize, const size_t dim, const size_t nProcessors, const size_t validCount) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto warpIdx = idx >> 5;
    if (warpIdx >= nProcessors) {
        return;
    }
    const auto laneIdx = idx & 31;
    const auto begin = warpIdx * (Iter << 5);
    auto outputBegin = warpIdx > 0 ? processorCounts[warpIdx - 1] : 0;

#pragma unroll
    for (auto i = 0; i < Iter; ++i) {
        const auto current = begin + (i << 5) + laneIdx;
        unsigned int localFlags = 0;
        bool localMask = false;
        if (current < dataSize) {
            localMask = mask[current];
            if (localMask) {
                localFlags = 1;
            }
        }
        // prefix Sum
        unsigned int temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 1);
        if (laneIdx >= 1) localFlags += temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 2);
        if (laneIdx >= 2) localFlags += temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 4);
        if (laneIdx >= 4) localFlags += temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 8);
        if (laneIdx >= 8) localFlags += temp;
        temp = __shfl_up_sync(0xFFFFFFFF, localFlags, 16);
        if (laneIdx >= 16) localFlags += temp;
        if (localMask) {
            const auto outputIdx = outputBegin + localFlags - 1;
            if (outputIdx >= validCount || current >= dataSize) {
                continue;
            }
            outputIndex[outputIdx] = inputIndex[current];
            for (size_t d = 0; d < dim; ++d) {
                outputData[outputIdx * dim + d] = inputData[current * dim + d];
            }
        }
        const auto numCurIter = __shfl_sync(0xFFFFFFFF, localFlags, 31);
        outputBegin += numCurIter;
    }
}

/**
 * @brief Compact a sparse grid based on a boolean mask.
 *
 * @param[in] grid The input sparse grid to be compacted.(device)
 * @param[in] mask A boolean array indicating which elements to keep.(host)
 * @param[in] validCount The number of true values in the mask.
 *
 * @return A new GridAsSparseMatrix containing only the elements where mask is true.(device)
 */
GridAsSparseMatrix* compactGrid(const GridAsSparseMatrix& grid, const bool* mask, const size_t validCount) {
    NvtxProfiler profiler("compactGrid", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Yellow);
    const auto len = grid.get_num_rows();
    const auto dim = grid.get_dimensions();
    const auto nnz = grid.get_nnz_nums();

    const auto* index = grid.get_ids_();
    const auto* data = grid.get_vals_();

    bool* d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, nnz * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_mask, mask, nnz * sizeof(bool), cudaMemcpyHostToDevice));

    // Step 1 : Count
    // blockSize : number of threads per block
    // Kp : number of elements processed by each warp
    constexpr int blockSize = 512;
    constexpr int Kp = 128;
    const auto nProcessors = (nnz + Kp - 1) / Kp;
    unsigned int* processorCounts;
    CUDA_CHECK(cudaMalloc(&processorCounts, nProcessors * sizeof(unsigned int)));
    launchCountKernel<bool, Kp, blockSize>(processorCounts, d_mask, nnz);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 2 : Prefix Scan
    launchPrefixSumKernel<unsigned int, blockSize>(processorCounts, nProcessors);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3 : Scatter
    constexpr auto warpPerBlock = blockSize >> 5;
    const auto gridSize = (nProcessors + warpPerBlock - 1) / warpPerBlock;
    constexpr auto Iter = Kp >> 5;

    size_t* d_compact_indexes;
    double* d_compact_data;
    CUDA_CHECK(cudaMalloc(&d_compact_indexes, validCount * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_compact_data, dim * validCount * sizeof(double)));

    // debug
    // nnz
    // dim
    // nProcessors
    // validCount
    // gridSize
    // blockSize
    double* d_data;
    size_t* d_index;

    cudaPointerAttributes dataAttr{};
    cudaPointerAttributes indexAttr{};
    CUDA_CHECK(cudaPointerGetAttributes(&dataAttr, data));
    if (dataAttr.type == cudaMemoryTypeDevice) {
        d_data = const_cast<double*>(data);
    } else {
        CUDA_CHECK(cudaMalloc(&d_data, nnz * dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, data, nnz * dim * sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaPointerGetAttributes(&indexAttr, index));
    if (indexAttr.type == cudaMemoryTypeDevice) {
        d_index = const_cast<size_t*>(index);
    } else {
        CUDA_CHECK(cudaMalloc(&d_index, nnz * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(d_index, index, nnz * sizeof(size_t), cudaMemcpyHostToDevice));
    }

    gridCompactPrefixKernel<Iter><<<gridSize, blockSize>>>(d_compact_data, d_compact_indexes, d_data, d_index, d_mask, processorCounts, nnz, dim, nProcessors, validCount);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(processorCounts));
    if (dataAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_data));
    }
    if (indexAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_index));
    }
    return new GridAsSparseMatrix{len, dim, validCount, d_compact_indexes, d_compact_data};
}
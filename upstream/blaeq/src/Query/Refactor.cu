//
// Created by Mgepahmge on 2025/12/19.
//

#include "Refactor.cuh"
#include "check.cuh"
#include "src/utils/NVTXProfiler.cuh"


__global__ void refactorKernel(size_t* data, const size_t* map, const size_t n) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n) {
        auto val = data[idx];
        val = map[val];
        data[idx] = val;
    }
}

/**
 * @brief Refactor a sparse grid based on a mapping array.
 *
 * @param[in, out] grid The sparse grid to be refactored. (device)
 * @param[in] map A mapping array indicating the new positions of the elements. (device)
 */
void refactor(GridAsSparseMatrix& grid, const size_t* map) {
    NvtxProfiler profiler("refactor", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Purple);
    const auto n = grid.get_nnz_nums();
    auto* index = const_cast<size_t*>(grid.get_ids_());
    dim3 block(256);
    dim3 gridDim((n + block.x - 1) / block.x);
    CHECK_MEM_POS(map, cudaMemoryTypeDevice);
    refactorKernel<<<gridDim, block>>>(index, map, n);
    CUDA_CHECK(cudaDeviceSynchronize());
}
//
// Created by Mgepahmge on 2025/12/18.
//

#include "RangePruning.cuh"
#include "check.cuh"
#include "src/utils/NVTXProfiler.cuh"
#include <algorithm>

__global__ void rangePruningKernel(bool* mask, const double* lowBounds, const double* upBounds, const double* centroids,
                                   const double* radius, const size_t dim, const size_t p, const size_t* indexs) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p) {
        double dist = 0.0;
        for (auto d = 0; d < dim; ++d) {
            double diff = 0.0;
            const auto c_val = centroids[idx * dim + d];
            const auto min_val = lowBounds[d];
            const auto max_val = upBounds[d];
            if (c_val < min_val) {
                diff = min_val - c_val;
            }
            else if (c_val > max_val) {
                diff = c_val - max_val;
            }
            dist += diff * diff;
        }
        const auto index = indexs[idx];
        if (const auto r = radius[index]; dist <= r * r) {
            mask[idx] = true;
        }
        else {
            mask[idx] = false;
        }
    }
}

/**
 * @brief Perform Range pruning.
 *
 * @param[in] lowBounds The lower bounds of the query range.(host/device)
 * @param[in] upBounds The upper bounds of the query range.(host/device)
 * @param[in] p The number of clusters for K-means labeling.
 * @param[in] dim The dimensionality of the data points.
 * @param[in] centroids The centroids of the K-means clusters.(host/device)
 * @param[in] radius The radius for each cluster.(host/device)
 * @param[in] indexs The original indexes of the data points.(host/device)
 * @param[in] length logic length of the grid.
 * @param[in,out] out_selected_count The number of selected clusters after pruning.
 *
 * @return A boolean array indicating which points are retained after pruning.(host)
 */
bool* rangePruning(const double* lowBounds, const double* upBounds, const size_t dim, const double* centroids,
                   const double* radius, const size_t p, const size_t* indexs, const size_t length, size_t& out_selected_count) {
    NvtxProfiler profiler("rangePruning", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Blue);
    auto* mask = new bool[p];
    bool* d_mask;
    CUDA_CHECK(cudaMalloc(&d_mask, p * sizeof(bool)));

    // Step 1 : Prepare device memory
    double* d_lowBounds;
    double* d_upBounds;
    double* d_centroids;
    double* d_radius;
    size_t* d_indexs;

    cudaPointerAttributes lowAttr{}, upAttr{}, centAttr{}, radAttr{}, idxAttr{};
    CUDA_CHECK(cudaPointerGetAttributes(&lowAttr, lowBounds));
    CUDA_CHECK(cudaPointerGetAttributes(&upAttr, upBounds));
    CUDA_CHECK(cudaPointerGetAttributes(&centAttr, centroids));
    CUDA_CHECK(cudaPointerGetAttributes(&radAttr, radius));
    CUDA_CHECK(cudaPointerGetAttributes(&idxAttr, indexs));

    if (lowAttr.type == cudaMemoryTypeDevice) {
        d_lowBounds = const_cast<double*>(lowBounds);
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_lowBounds, dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_lowBounds, lowBounds, dim * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (upAttr.type == cudaMemoryTypeDevice) {
        d_upBounds = const_cast<double*>(upBounds);
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_upBounds, dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_upBounds, upBounds, dim * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (centAttr.type == cudaMemoryTypeDevice) {
        d_centroids = const_cast<double*>(centroids);
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_centroids, p * dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids, p * dim * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (radAttr.type == cudaMemoryTypeDevice) {
        d_radius = const_cast<double*>(radius);
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_radius, length * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_radius, radius, length * sizeof(double), cudaMemcpyHostToDevice));
    }

    if (idxAttr.type == cudaMemoryTypeDevice) {
        d_indexs = const_cast<size_t*>(indexs);
    }
    else {
        CUDA_CHECK(cudaMalloc(&d_indexs, p * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(d_indexs, indexs, p * sizeof(size_t), cudaMemcpyHostToDevice));
    }

    // Step 2 : Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x);
    rangePruningKernel<<<gridSize, blockSize>>>(d_mask, d_lowBounds, d_upBounds, d_centroids, d_radius, dim, p, d_indexs);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (lowAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_lowBounds));
    }
    if (upAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_upBounds));
    }
    if (centAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_centroids));
    }
    if (radAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_radius));
    }
    if (idxAttr.type != cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(d_indexs));
    }

    CUDA_CHECK(cudaMemcpy(mask, d_mask, p * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mask));
    out_selected_count = std::count(mask, mask + p, true);
    return mask;
}
//
// Created by Mgepahmge on 2025/12/12.
//
#include "KNN.cuh"
#include "src/MergeSort/MergeSort.cuh"
#include <algorithm>
#include "check.cuh"
#include "src/utils/NVTXProfiler.cuh"

struct Cluster {
    double distance;
    uint64_t label;

    __device__ __host__ bool operator<(const Cluster& other) const {
        return distance < other.distance;
    }

    __device__ __host__ bool operator>(const Cluster& other) const {
        return distance > other.distance;
    }

    __device__ __host__ bool operator==(const Cluster& other) const {
        return distance == other.distance;
    }

    __device__ __host__ bool operator<=(const Cluster& other) const {
        return distance <= other.distance;
    }

    __device__ __host__ bool operator>=(const Cluster& other) const {
        return distance >= other.distance;
    }
};

__global__ void calculateClusterDistanceKernel(Cluster* clusters, const double* query_point, const double* centroids, const double* radius, const size_t dim, const size_t* indexs, const size_t p) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < p) {
        const auto index = indexs[idx];
        clusters[idx].label = idx;
        double dist = 0.0;
        for (size_t d = 0; d < dim; ++d) {
            const auto diff = query_point[d] - centroids[idx * dim + d];
            dist += diff * diff;
        }
        dist = __dsqrt_rd(dist);
        clusters[idx].distance = dist - radius[index];
    }
}

/**
 * @brief Perform k-NN pruning.
 *
 * @details using STEP algorithm.
 *
 * @param[in] k The number of nearest neighbors to consider.
 * @param[in] p The number of clusters for K-means labeling.
 * @param[in] dim The dimensionality of the data points.
 * @param[in] length logic length of the grid.
 * @param[in] query_point The query point for which to perform pruning.(host/device)
 * @param[in] centroids The centroids of the K-means clusters.(host/device)
 * @param[in] radius The radius for each cluster.(host/device)
 * @param[in] cluster_sizes The sizes of each cluster.(host/device)
 * @param[in] indexs The original indexes of the data points.(host/device)
 * @param[in,out] out_selected_count The number of selected clusters after pruning.
 *
 * @return A boolean array indicating which points are retained after pruning.(host)
 */
bool* knnPruning(const size_t k, const size_t p, const size_t dim, const size_t length, const double* query_point,
                 const double* centroids, const double* radius,
                 const size_t* cluster_sizes, const size_t* indexs, size_t& out_selected_count) {
    NvtxProfiler profiler("knnPruning", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Blue);

    NvtxProfiler innerProfiler1("准备数据", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Cyan);

    auto* result = new bool[p];

    double* d_centroids = nullptr;
    double* d_radius = nullptr;
    double* h_radius = nullptr;
    double* d_query_point = nullptr;
    size_t* h_cluster_sizes = nullptr;
    size_t* d_indexs = nullptr;
    size_t* h_indexs = nullptr;

    cudaPointerAttributes centroids_attr{};
    cudaPointerAttributes radius_attr{};
    cudaPointerAttributes query_point_attr{};
    cudaPointerAttributes cluster_sizes_attr{};
    cudaPointerAttributes indexs_attr{};

    CUDA_CHECK(cudaPointerGetAttributes(&centroids_attr, centroids));
    CUDA_CHECK(cudaPointerGetAttributes(&radius_attr, radius));
    CUDA_CHECK(cudaPointerGetAttributes(&query_point_attr, query_point));
    CUDA_CHECK(cudaPointerGetAttributes(&cluster_sizes_attr, cluster_sizes));
    CUDA_CHECK(cudaPointerGetAttributes(&indexs_attr, indexs));

    Cluster* d_clusters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_clusters, p * sizeof(Cluster)));

    if (centroids_attr.type == cudaMemoryTypeHost || centroids_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaMalloc(&d_centroids, p * dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids, p * dim * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        d_centroids = const_cast<double*>(centroids);
    }

    if (radius_attr.type == cudaMemoryTypeHost || radius_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaMalloc(&d_radius, length * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_radius, radius, length * sizeof(double), cudaMemcpyHostToDevice));
        h_radius = const_cast<double*>(radius);
    } else {
        h_radius = new double[length];
        CUDA_CHECK(cudaMemcpy(h_radius, radius, length * sizeof(double), cudaMemcpyDeviceToHost));
        d_radius = const_cast<double*>(radius);
    }

    if (query_point_attr.type == cudaMemoryTypeHost || query_point_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaMalloc(&d_query_point, dim * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_query_point, query_point, dim * sizeof(double), cudaMemcpyHostToDevice));
    } else {
        d_query_point = const_cast<double*>(query_point);
    }

    if (indexs_attr.type == cudaMemoryTypeHost || indexs_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaMalloc(&d_indexs, p * sizeof(size_t)));
        CUDA_CHECK(cudaMemcpy(d_indexs, indexs, p * sizeof(size_t), cudaMemcpyHostToDevice));
        h_indexs = const_cast<size_t*>(indexs);
    } else {
        d_indexs = const_cast<size_t*>(indexs);
        h_indexs = new size_t[p];
        CUDA_CHECK(cudaMemcpy(h_indexs, indexs, p * sizeof(size_t), cudaMemcpyDeviceToHost));
    }

    innerProfiler1.release();

    NvtxProfiler innerProfiler4("计算距离", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::SkyBlue);
    dim3 blockSize(256);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x);
    calculateClusterDistanceKernel<<<gridSize, blockSize>>>(d_clusters, d_query_point, d_centroids, d_radius, dim, d_indexs, p);
    CUDA_CHECK(cudaDeviceSynchronize());
    innerProfiler4.release();

    NvtxProfiler innerProfiler5("排序", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::SpringGreen);
    constexpr int IPB = 512;
    constexpr int IPT = 2;
    Cluster cluster_max{};
    cluster_max.distance = std::numeric_limits<double>::max();
    cluster_max.label = 0;
    gpuSort<IPB, IPT, true>(d_clusters, d_clusters, p, cluster_max);

    auto* h_clusters = new Cluster[p];

    if (cluster_sizes_attr.type == cudaMemoryTypeHost || cluster_sizes_attr.type == cudaMemoryTypeUnregistered) {
        h_cluster_sizes = const_cast<size_t*>(cluster_sizes);
    } else {
        h_cluster_sizes = new size_t[p];
        CUDA_CHECK(cudaMemcpy(h_cluster_sizes, cluster_sizes, p * sizeof(size_t), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    innerProfiler5.release();

    CUDA_CHECK(cudaMemcpy(h_clusters, d_clusters, p * sizeof(Cluster), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_clusters));
    if (centroids_attr.type == cudaMemoryTypeHost || centroids_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaFree(d_centroids));
    }
    if (radius_attr.type == cudaMemoryTypeHost || radius_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaFree(d_radius));
    }
    if (query_point_attr.type == cudaMemoryTypeHost || query_point_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaFree(d_query_point));
    }
    if (indexs_attr.type == cudaMemoryTypeHost || indexs_attr.type == cudaMemoryTypeUnregistered) {
        CUDA_CHECK(cudaFree(d_indexs));
    }

    NvtxProfiler innerProfiler2("剪枝", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    // Select top-k clusters
    size_t current_count = 0;
    double current_distance = 0.0;
    std::vector<size_t> selected_labels;
    size_t selected_count = 0;

    for (auto i = 0; i < p; ++i) {
        if (current_count >= k) {
            if (h_clusters[i].distance > current_distance) {
                break;
            }
            const auto index = h_indexs[h_clusters[i].label];
            current_count += h_cluster_sizes[index];
            selected_labels.push_back(h_clusters[i].label);
            ++selected_count;
        } else {
            const auto index = h_indexs[h_clusters[i].label];
            current_count += h_cluster_sizes[index];
            current_distance = std::max(current_distance, h_clusters[i].distance + 2 * h_radius[index]);
            selected_labels.push_back(h_clusters[i].label);
            ++selected_count;
        }
    }

    innerProfiler2.release();
    NvtxProfiler innerProfiler3("清理数据与准备结果", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Magenta);

    delete[] h_clusters;
    if (cluster_sizes_attr.type != cudaMemoryTypeHost && cluster_sizes_attr.type != cudaMemoryTypeUnregistered) {
        delete[] h_cluster_sizes;
    }
    if (radius_attr.type != cudaMemoryTypeHost && radius_attr.type != cudaMemoryTypeUnregistered) {
        delete[] h_radius;
    }
    if (indexs_attr.type != cudaMemoryTypeHost && indexs_attr.type != cudaMemoryTypeUnregistered) {
        delete[] h_indexs;
    }

    std::fill(result, result + p, false);

    for (const auto& label : selected_labels) {
        result[label] = true;
    }

    out_selected_count = selected_count;
    return result;
}
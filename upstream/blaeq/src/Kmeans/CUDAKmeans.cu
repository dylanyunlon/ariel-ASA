#include "CUDAKmeans.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <algorithm>

CUDAKmeans::CUDAKmeans(const double* data, size_t N, size_t dim, bool is_aos)
    : N_(N), dim_(dim), k_(0) {

    handle_ = std::make_unique<raft::resources>();

    datas_.reserve(N);
    labels_.resize(N);

    convertToPointVector(data, N, dim, is_aos);

    uploadDataToGPU(data, N, dim, is_aos);
}

CUDAKmeans::~CUDAKmeans() {
}

void CUDAKmeans::convertToPointVector(const double* flat_data, size_t N, 
                                      size_t dim, bool is_aos) {
    datas_.clear();
    datas_.reserve(N);
    
    if (is_aos) {
        // Array of Structures: [p0_d0, p0_d1, ..., p1_d0, p1_d1, ...]
        for (size_t i = 0; i < N; ++i) {
            Point p(dim);
            for (size_t d = 0; d < dim; ++d) {
                p[d] = flat_data[i * dim + d];
            }
            datas_.push_back(std::move(p));
        }
    } else {
        // Structure of Arrays: [p0_d0, p1_d0, ..., p0_d1, p1_d1, ...]
        for (size_t i = 0; i < N; ++i) {
            Point p(dim);
            for (size_t d = 0; d < dim; ++d) {
                p[d] = flat_data[d * N + i];
            }
            datas_.push_back(std::move(p));
        }
    }
}

void CUDAKmeans::uploadDataToGPU(const double* data, size_t N, size_t dim, bool is_aos) {
    dataset_gpu_ = std::make_unique<raft::device_matrix<double, int>>(
        raft::make_device_matrix<double, int>(*handle_, N, dim));
    
    if (is_aos) {
        cudaMemcpy(dataset_gpu_->data_handle(), data, N * dim * sizeof(double),
                   cudaMemcpyHostToDevice);
    } else {
        std::vector<double> temp(N * dim);
        for (size_t i = 0; i < N; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                temp[i * dim + d] = data[d * N + i];
            }
        }
        cudaMemcpy(dataset_gpu_->data_handle(), temp.data(), 
                   N * dim * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void CUDAKmeans::run(size_t k, size_t max_iters) {
    k_ = k;

    centroids_gpu_ = std::make_unique<raft::device_matrix<double, int>>(
        raft::make_device_matrix<double, int>(*handle_, k, dim_));
    labels_gpu_ = std::make_unique<raft::device_vector<int, int>>(
        raft::make_device_vector<int, int>(*handle_, N_));

    cuvs::cluster::kmeans::params kmeans_params;
    kmeans_params.n_clusters = k;
    kmeans_params.max_iter = max_iters;
    kmeans_params.tol = 0.0001;
    kmeans_params.metric = cuvs::distance::DistanceType::L2Expanded;
    kmeans_params.init = cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus;

    auto dataset_view = raft::make_device_matrix_view<const double, int>(
        dataset_gpu_->data_handle(), N_, dim_);

    double inertia = 0.0;
    int n_iter = 0;

    cuvs::cluster::kmeans::fit(
        *handle_,
        kmeans_params,
        dataset_view,
        std::nullopt,  // sample_weight
        centroids_gpu_->view(),
        raft::make_host_scalar_view(&inertia),
        raft::make_host_scalar_view(&n_iter)
    );

    cuvs::cluster::kmeans::predict(
        *handle_,
        kmeans_params,
        dataset_view,
        std::nullopt,  // sample_weight
        centroids_gpu_->view(),
        labels_gpu_->view(),
        true,  // normalize_weight
        raft::make_host_scalar_view(&inertia)
    );

    cudaDeviceSynchronize();
    downloadResultsFromGPU();
}

void CUDAKmeans::downloadResultsFromGPU() {
    std::vector<int> temp_labels(N_);
    cudaMemcpy(temp_labels.data(), labels_gpu_->data_handle(), 
               N_ * sizeof(int), cudaMemcpyDeviceToHost);
    
    labels_.resize(N_);
    for (size_t i = 0; i < N_; ++i) {
        labels_[i] = static_cast<size_t>(temp_labels[i]);
    }

    std::vector<double> temp_centroids(k_ * dim_);
    cudaMemcpy(temp_centroids.data(), centroids_gpu_->data_handle(),
               k_ * dim_ * sizeof(double), cudaMemcpyDeviceToHost);
    
    centroids_.clear();
    centroids_.reserve(k_);
    for (size_t c = 0; c < k_; ++c) {
        Point p(dim_);
        for (size_t d = 0; d < dim_; ++d) {
            p[d] = temp_centroids[c * dim_ + d];
        }
        centroids_.push_back(std::move(p));
    }
}

void CUDAKmeans::displayGroup() {
    std::vector<std::vector<size_t>> groups(k_);
    
    printf("dataset has %ld rows\n", N_);
    printf("then will be divided into %ld parts\n", k_);

    for (size_t i = 0; i < labels_.size(); ++i) {
        groups[labels_[i]].push_back(i);
    }

    for (size_t c = 0; c < groups.size(); ++c) {
        printf("Group %ld contains %ld elements\n", c, groups[c].size());

        const Point& center = centroids_[c];
        std::cout << "current center point is (";
        for (size_t d = 0; d < center.size(); ++d) {
            std::cout << center[d];
            if (d < center.size() - 1) std::cout << " ";
        }
        std::cout << ")" << std::endl;

        int limit = 5;
        for (auto idx : groups[c]) {
            if (limit <= 0) {
                std::cout << std::endl;
                break;
            }
            const Point& p = datas_[idx];
            for (auto val : p) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            limit--;
        }
    }
}

void CUDAKmeans::reset() {
    datas_ = std::move(centroids_);
    N_ = datas_.size();
    
    labels_.clear();
    labels_.resize(N_);
    centroids_.clear();
    k_ = 0;

    std::vector<double> flat_data(N_ * dim_);
    for (size_t i = 0; i < N_; ++i) {
        for (size_t d = 0; d < dim_; ++d) {
            flat_data[i * dim_ + d] = datas_[i][d];
        }
    }
    
    uploadDataToGPU(flat_data.data(), N_, dim_, true);
}

void CUDAKmeans::clean_and_reset(const double* flat_data, size_t N, size_t dim) {
    datas_.clear();
    labels_.clear();
    centroids_.clear();
    
    N_ = N;
    dim_ = dim;
    k_ = 0;
    
    datas_.reserve(N);
    labels_.resize(N);

    convertToPointVector(flat_data, N, dim, false);
    uploadDataToGPU(flat_data, N, dim, false);
}
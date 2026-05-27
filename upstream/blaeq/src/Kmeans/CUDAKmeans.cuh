#ifndef CUDA_KMEANS_H
#define CUDA_KMEANS_H

#include <vector>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <cuvs/cluster/kmeans.hpp>

class CUDAKmeans {
public:
    using Point = std::vector<double>;

    CUDAKmeans(const double* data, size_t N, size_t dim, bool is_aos = true);

    ~CUDAKmeans();

    void run(size_t k, size_t max_iters = 100);

    void displayGroup();

    void reset();

    void clean_and_reset(const double* flat_data, size_t N, size_t dim);

    [[nodiscard]] const std::vector<size_t>& getLabels() const { return labels_; }
    [[nodiscard]] const std::vector<Point>& getCentroids() const { return centroids_; }
    [[nodiscard]] const std::vector<Point>& getdatas() const { return datas_; }
    [[nodiscard]] size_t get_curr_layer_length() const { return datas_.size(); }
    [[nodiscard]] size_t get_next_layer_length() const { return centroids_.size(); }

private:
    std::vector<Point> datas_;
    std::vector<size_t> labels_;
    std::vector<Point> centroids_;

    std::unique_ptr<raft::resources> handle_;

    size_t N_;    // 数据点数量
    size_t dim_;  // 数据维度
    size_t k_;    // 簇数量

    std::unique_ptr<raft::device_matrix<double, int>> dataset_gpu_;
    std::unique_ptr<raft::device_matrix<double, int>> centroids_gpu_;
    std::unique_ptr<raft::device_vector<int, int>> labels_gpu_;

    void uploadDataToGPU(const double* data, size_t N, size_t dim, bool is_aos);
    void downloadResultsFromGPU();
    void convertToPointVector(const double* flat_data, size_t N, size_t dim, bool is_aos);
};

#endif // CUDA_KMEANS_H
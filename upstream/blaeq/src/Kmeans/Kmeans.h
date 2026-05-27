#ifndef KMEANS_H
#define KMEANS_H
#pragma once
#include <vector>
#include <cstddef>

class KMeans {
public:
    using Point = std::vector<double>;

    KMeans(const double* data, size_t N, size_t dim, bool is_aos);
    void run(size_t k, size_t max_iters = 50);
    void displayGroup();
    void reset();
    void clean_and_reset(const double* flat_data, size_t N, size_t dim);

    const std::vector<size_t>& getLabels()    const { return labels_; }
    const std::vector<Point>&  getCentroids() const { return centroids_; }
    const std::vector<Point>&  getdatas()     const { return datas_; }
    size_t get_curr_layer_length() const { return datas_.size(); }
    size_t get_next_layer_length() const { return centroids_.size(); }

private:
    std::vector<Point>  datas_;
    std::vector<size_t> labels_;
    std::vector<Point>  centroids_;

    void initCentroids(size_t k);
    bool assignStep();
    void updateStep();


};

#endif // KMEANS_H
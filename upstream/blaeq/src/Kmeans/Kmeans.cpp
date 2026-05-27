#include <random>
#include <limits>
#include <numeric>
#include <iostream>
#include <stdio.h>
#include "Kmeans.h"

using namespace std;

/* calculate the distance of two points !!!*/
namespace {
double euclid2(const KMeans::Point& a, const KMeans::Point& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}
} // namespace

/* ---------- constructor ---------- */
/*
flat_data : the pointer to the dataset array
N :   row nums
dim : dimensional nums
*/
KMeans::KMeans(const double* flat_data, size_t N, size_t dim, bool is_aos) {
    // common
    datas_.reserve(N);

    if(is_aos){
        for(size_t i = 0; i < N; i++){
            Point p(dim);
            for(size_t d = 0; d < dim; d++) p[d] = flat_data[i * dim + d];
            datas_.push_back(std::move(p));
        }
    }

    else{
        for(size_t i = 0; i < N; i++) {
            Point p(dim);
            // xak review : based on the flat_data format to modify this code !!!
            for(size_t d = 0; d < dim; d++) p[d] = flat_data[d * N + i];
            datas_.push_back(std::move(p));
        }
    }



    labels_.resize(N);
}

/* ---------- 主入口 ---------- */
void KMeans::run(size_t k, size_t max_iters) {
    initCentroids(k);
    for (size_t iter = 0; iter < max_iters; ++iter) {
        if (!assignStep()) break;
        updateStep();
    }
}

/* ---------- k-means++ 初始化 ---------- */
void KMeans::initCentroids(size_t k) {
    centroids_.clear();
    centroids_.reserve(k);
    random_device rd;
    mt19937 gen(rd());
    // data_.size() == N
    uniform_int_distribution<size_t> dis(0, datas_.size() - 1);

    // random pick one center
    centroids_.push_back(datas_[dis(gen)]);

    vector<double> dist2(datas_.size(), numeric_limits<double>::max());
    // dist2 : one point to all center's min distance
    // pick k-1 center points
    for (size_t c = 1; c < k; ++c) {
        double sum = 0;
        // loop all points
        // pick current last center, get sigma d(point, last_center)
        for (size_t i = 0; i < datas_.size(); ++i) {
            double d = euclid2(datas_[i], centroids_.back());
            if (d < dist2[i]) dist2[i] = d;
            sum += dist2[i];
        }
        uniform_real_distribution<double> u(0, sum);
        double thresh = u(gen);
        double cum = 0;
        for (size_t i = 0; i < datas_.size(); ++i) {
            cum += dist2[i];
            //tend to pick far point !
            if (cum >= thresh) {
                centroids_.push_back(datas_[i]);
                break;
            }
        }
    }
}

/* based on current k center points, rearrage the range */
bool KMeans::assignStep() {
    bool changed = false;
    for (size_t i = 0; i < datas_.size(); ++i) {
        double best_dist = numeric_limits<double>::max();
        size_t best_id = 0;
        for (size_t c = 0; c < centroids_.size(); ++c) {
            double d = euclid2(datas_[i], centroids_[c]);
            if (d < best_dist) { best_dist = d; best_id = c; }
        }
        // get best_dist and best_id
        if (labels_[i] != best_id) { labels_[i] = best_id; changed = true; }
    }
    return changed;
}

/* update, based on a new range, readjust the center points */
void KMeans::updateStep() {
    const size_t k   = centroids_.size();
    const size_t dim = datas_[0].size();
    vector<Point> new_cent(k, Point(dim, 0.0));
    vector<size_t> count(k, 0);

    for (size_t i = 0; i < datas_.size(); ++i) {
        size_t c = labels_[i];
        //point ++
        //count ++
        for (size_t d = 0; d < dim; ++d) new_cent[c][d] += datas_[i][d];
        count[c]++;
    }
    for (size_t c = 0; c < k; ++c) {
        if (count[c] == 0) continue;
        for (size_t d = 0; d < dim; ++d) new_cent[c][d] /= count[c];
        centroids_[c] = std::move(new_cent[c]);
    }
}

/* display all center groups*/
void KMeans::displayGroup(){
    std::vector<std::vector<size_t>> vec;
    const size_t D = datas_.size();
    printf("dataset has %ld rows\n", D);
    const size_t K = centroids_.size();
    printf("then will be divided into %ld parts\n", K);
    vec.resize(K);
    for(size_t t = 0; t < labels_.size(); t++){
        size_t c = labels_[t];
        vec[c].push_back(t);
    }
    for(size_t t = 0; t < vec.size(); t++){
        size_t element_nums = vec[t].size();
        //print group
        printf("Group %ld contains %ld elements\n", t, element_nums);
        //print current center point
        {
            Point& curr_center = centroids_[t];
            std::cout << "current center point is (";
            int p = curr_center.size();
            p--;
            for(auto &curr_center_dim : curr_center){
                std::cout << curr_center_dim;
                if(p == 0) break;
                std::cout << " ";
                p--;
            }
            std:: cout << ") " << std::endl;
        }
        int limit = 5;
        for(auto &element_id : vec[t]){
            if(limit <= 0) { std::cout << std::endl; break; }
            Point& curr_point = datas_[element_id];
            for(auto &element_dim : curr_point){
                std::cout << element_dim << " ";
            }
            std::cout << std::endl;
            limit--;
        }
    }
}

// KMeans member func :  to clear all the data in class
void KMeans::reset()
{   
    datas_ = std::move(centroids_);
    labels_.clear();
    centroids_.clear();
    labels_.resize(datas_.size());
}

// Kmeans member func : to clear and refulfill the data
void KMeans::clean_and_reset(const double* flat_data, size_t N, size_t dim){
    datas_.clear();
    labels_.clear();
    centroids_.clear();
    datas_.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        Point p(dim);
        // xak review : based on the flat_data format to modify this code !!!
        for (size_t d = 0; d < dim; ++d) p[d] = flat_data[d * N + i];
        datas_.push_back(std::move(p));
    }
    labels_.resize(N);
}
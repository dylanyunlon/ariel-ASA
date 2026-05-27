#pragma once
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <stdexcept>
#include <iomanip> 
#include <algorithm>   // std::sort, std::iota
#include <numeric>
#include <cassert>
#include <fstream>
namespace Comp {

//  |val| < 1e-6
inline bool isZero(double val) {
    static constexpr double EPSILON = 1e-6;
    return std::fabs(val) < EPSILON;
}

// define epsilon by your own 
inline bool isZeroWithEps(double val, double epsilon) {
    return std::fabs(val) < epsilon;
}

// vec comp
inline bool isVectorEqual(const std::vector<double>& v1,
                          const std::vector<double>& v2) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i)
        if (!isZero(v1[i] - v2[i])) return false;
    return true;
}

// mat comp
inline bool isMatrixEqual(const std::vector<std::vector<double>>& m1,
                          const std::vector<std::vector<double>>& m2) {
    if (m1.size() != m2.size()) return false;
    if (m1.empty()) return true;
    const size_t cols = m1[0].size();
    for (const auto& row : m2)
        if (row.size() != cols) return false;

    for (size_t i = 0; i < m1.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            if (!isZero(m1[i][j] - m2[i][j])) return false;
    return true;
}

} // namespace Comp




namespace RandomSelector {

// 按概率 p 抽取 raw 中的元素，返回新 vector
inline std::vector<size_t> generate_with_probability_random_size_t_vec(double p, const std::vector<size_t>& raw)
{
    if (p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be in [0,1]");

    std::vector<size_t> result;
    std::bernoulli_distribution dist(p);
    std::mt19937 rng{std::random_device{}()};   // 只构造一次，性能更好

    result.reserve(raw.size());                 // 提前 reserve 避免反复分配
    for (size_t elem : raw)
        if (dist(rng)) result.push_back(elem);

    return result;                              // 空 vector 自然 size()==0
}

} // namespace RandomSelector

namespace Simulation {

// print process：[====>      ] 40.0% (40/100)
inline void showProgress(int current, int total)
{
    if (total <= 0) return;               // 防止除 0
    const float progress = static_cast<float>(current) / total;
    constexpr int barWidth = 50;

    std::cout << '\r' << '[';             // 回行首
    const int pos = static_cast<int>(barWidth * progress);

    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)      std::cout << '=';
        else if (i == pos) std::cout << '>';
        else              std::cout << ' ';
    }

    std::cout << "] " << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "% "
              << '(' << current << '/' << total << ')' << std::flush;

    std::cout << std::endl;
}

} // namespace Simulation

namespace Sort {

// 返回排序后的“下标数组”：labels[result[i]] 按 ascending/descending 有序
inline std::vector<size_t> Sorted_Layer_With_Original_idxs(const std::vector<size_t>& labels, bool ascending = true)
{
    assert(!labels.empty());
    std::vector<size_t> idx(labels.size());
    std::iota(idx.begin(), idx.end(), 0);          // 0,1,2,...,n-1
    std::sort(idx.begin(), idx.end(),
              [&labels, ascending](size_t i, size_t j)
              {
                  return ascending ? labels[i] < labels[j]
                                   : labels[i] > labels[j];
              });
    return idx;
}

} // namespace Sort

namespace Chrono {

/*
 * 自适应打印 t1 -> t2 的间隔：
 *  >= 1 s   ->  1.234 s
 *  >= 1 ms  ->  12.34 ms
 *  otherwise->  123.4 µs
 */
template<class Clock = std::chrono::steady_clock>
void printElapsed(const std::string& str,
                  const typename Clock::time_point& t1,
                  const typename Clock::time_point& t2)
{
    using namespace std::chrono;
    auto dur = t2 - t1;

    std::cout << str << " : ";
    if (dur >= duration<double>(1))                    // ≥ 1 s
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double>(dur).count() << " s\n";
    else if (dur >= duration<double, std::milli>(1))   // ≥ 1 ms
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::milli>(dur).count() << " ms\n";
    else                                               // < 1 ms
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::micro>(dur).count() << " µs\n";
}

template<class Clock = std::chrono::steady_clock>
void printAvgElapsed(const std::string& str,
                  const typename Clock::time_point& t1,
                  const typename Clock::time_point& t2,
                  size_t total_times)
{
    using namespace std::chrono;
    if (total_times == 0) {
        return;
    }          // 防止除 0
    auto dur = (t2 - t1) / total_times;    // 平均时长

    std::cout << str << " : ";
    if (dur >= duration<double>(1))                    // ≥ 1 s
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double>(dur).count() << " s\n";
    else if (dur >= duration<double, std::milli>(1))   // ≥ 1 ms
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::milli>(dur).count() << " ms\n";
    else                                               // < 1 ms
        std::cout << std::fixed << std::setprecision(3)
                  << duration<double, std::micro>(dur).count() << " µs\n";
}

} // namespace Chrono


// namespace dist meric
namespace dist {

// euclidean distance
// calc the euclidiean distance between vec a and vec b
inline double euclidean(const std::vector<double>& a,
                        const std::vector<double>& b)
{
    // 1. dim must be the same !!!
    if (a.size() != b.size())
    {
        assert(false);
        throw std::invalid_argument("Size mismatch in euclidean().");
    }

    // 2. foreach dim
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// the square of euclidiean distance
inline double euclidean_squared(const std::vector<double>& a,
                                const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("Size mismatch in euclidean_squared().");

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    // without sqrt
    return sum;   

}

}  // namespace dist


namespace Serialization{
inline void write_max_radius(std::string& base_dir,
                             size_t idx,
                             const double* max_radius,
                             std::size_t Centroids_nums)
{
    // 1. handle base_dir, to prevent corner case.
    if (!base_dir.empty() && base_dir.back() != '/') base_dir += '/';

    // 2 construct file name
    std::ostringstream oss;
    oss << base_dir << "Mesh_" << idx;

    /* 3. 打开文件并写入 */
    std::ofstream fout(oss.str());
    if (!fout.is_open())
        throw std::runtime_error("Load_To_File: cannot create " + oss.str());

    fout << std::fixed << std::setprecision(6);
    
    for (std::size_t i = 0; i < Centroids_nums; ++i) {
        fout << i << " : " << max_radius[i] << '\n';
    }
}


} // end namespace Serialization
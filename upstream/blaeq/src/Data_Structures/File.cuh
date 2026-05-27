//
// Created by cuda01 on 2025/12/26.
//

#ifndef BLAEQ_CUDA_FILE_CUH
#define BLAEQ_CUDA_FILE_CUH
#include "src/Data_Structures/Data_Structures.cuh"

Multidimensional_Arr loadFromFile(const std::string& filename);

void saveToFile(const Multidimensional_Arr& arr, const std::string& filename);

// 查询类型枚举
enum class QueryType {
    POINT,      // 查询点
    RANGE       // 查询范围
};

// 查询数据结构
class Query {
public:
    int length;           // 查询数量
    int dim;              // 维度
    QueryType type;       // 查询类型
    std::vector<double> data;  // 数据存储

    // 构造函数 - 查询点
    Query(int len, int d, QueryType t = QueryType::POINT)
        : length(len), dim(d), type(t) {
        if (type == QueryType::POINT) {
            data.resize(length * dim);
        } else {
            data.resize(2 * length * dim);
        }
    }

    // 获取一条查询点（返回单个数组）
    std::vector<double> getQueryPoint(int index) const {
        if (type != QueryType::POINT) {
            throw std::runtime_error("Cannot get query point from range query");
        }
        if (index < 0 || index >= length) {
            throw std::out_of_range("Query index out of range");
        }

        std::vector<double> point(dim);
        for (int i = 0; i < dim; i++) {
            point[i] = data[index * dim + i];
        }
        return point;
    }

    // 获取一条查询范围（返回下界和上界两个数组）
    std::pair<std::vector<double>, std::vector<double>> getQueryRange(int index) const {
        if (type != QueryType::RANGE) {
            throw std::runtime_error("Cannot get query range from point query");
        }
        if (index < 0 || index >= length) {
            throw std::out_of_range("Query index out of range");
        }

        std::vector<double> lower(dim);
        std::vector<double> upper(dim);

        int offset = index * 2 * dim;
        for (int i = 0; i < dim; i++) {
            lower[i] = data[offset + i];
            upper[i] = data[offset + dim + i];
        }

        return {lower, upper};
    }

    // 设置查询点
    void setQueryPoint(int index, const std::vector<double>& point) {
        if (type != QueryType::POINT) {
            throw std::runtime_error("Cannot set query point in range query");
        }
        if (index < 0 || index >= length) {
            throw std::out_of_range("Query index out of range");
        }
        if (point.size() != dim) {
            throw std::invalid_argument("Point dimension mismatch");
        }

        for (int i = 0; i < dim; i++) {
            data[index * dim + i] = point[i];
        }
    }

    // 设置查询范围
    void setQueryRange(int index, const std::vector<double>& lower,
                       const std::vector<double>& upper) {
        if (type != QueryType::RANGE) {
            throw std::runtime_error("Cannot set query range in point query");
        }
        if (index < 0 || index >= length) {
            throw std::out_of_range("Query index out of range");
        }
        if (lower.size() != dim || upper.size() != dim) {
            throw std::invalid_argument("Range dimension mismatch");
        }

        int offset = index * 2 * dim;
        for (int i = 0; i < dim; i++) {
            data[offset + i] = lower[i];
            data[offset + dim + i] = upper[i];
        }
    }
};

Query loadQueryPointFromFile(const std::string& filename);

void saveQueryPointToFile(const Query& query, const std::string& filename);

Query loadQueryRangeFromFile(const std::string& filename);

void saveQueryRangeToFile(const Query& query, const std::string& filename);

#endif //BLAEQ_CUDA_FILE_CUH
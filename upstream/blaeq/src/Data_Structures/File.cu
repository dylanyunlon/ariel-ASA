//
// Created by cuda01 on 2025/12/26.
//

#include "File.cuh"
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

Multidimensional_Arr loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int length = 0, dim = 0;
    std::string line;

    if (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> length >> dim)) {
            throw std::runtime_error("File format error: invalid metadata format");
        }

        if (length <= 0 || dim <= 0) {
            throw std::runtime_error("File format error: length and dimension must be positive");
        }
    } else {
        throw std::runtime_error("File format error: empty file");
    }

    Multidimensional_Arr arr(length, dim);

    int row = 0;
    while (std::getline(file, line) && row < length) {
        std::istringstream iss(line);
        double value;
        int col = 0;

        while (iss >> value && col < dim) {
            arr.data[row * dim + col] = value;
            col++;
        }

        if (col != dim) {
            throw std::runtime_error("File format error: dimension mismatch at row " +
                                   std::to_string(row + 1));
        }
        row++;
    }

    if (row != length) {
        throw std::runtime_error("File format error: row count mismatch");
    }

    file.close();
    return arr;
}

void saveToFile(const Multidimensional_Arr& arr, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create file: " + filename);
    }

    file << arr.N << " " << arr.D << "\n";

    for (int i = 0; i < arr.N; i++) {
        for (int j = 0; j < arr.D; j++) {
            file << arr.data[i * arr.D + j];
            if (j < arr.D - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
}

// 加载查询点文件
Query loadQueryPointFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int length = 0, dim = 0;
    std::string line;

    // 读取第一行：长度 维度
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> length >> dim)) {
            throw std::runtime_error("File format error: invalid metadata format");
        }

        if (length <= 0 || dim <= 0) {
            throw std::runtime_error("File format error: length and dimension must be positive");
        }
    } else {
        throw std::runtime_error("File format error: empty file");
    }

    Query query(length, dim, QueryType::POINT);

    int row = 0;
    while (std::getline(file, line) && row < length) {
        std::istringstream iss(line);
        double value;
        int col = 0;

        while (iss >> value && col < dim) {
            query.data[row * dim + col] = value;
            col++;
        }

        if (col != dim) {
            throw std::runtime_error("File format error: dimension mismatch at row " +
                                   std::to_string(row + 1));
        }
        row++;
    }

    if (row != length) {
        throw std::runtime_error("File format error: row count mismatch");
    }

    file.close();
    return query;
}

// 保存查询点到文件
void saveQueryPointToFile(const Query& query, const std::string& filename) {
    if (query.type != QueryType::POINT) {
        throw std::runtime_error("Can only save point queries with this function");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create file: " + filename);
    }

    // 第一行：长度 维度
    file << query.length << " " << query.dim << "\n";

    // 数据部分
    for (int i = 0; i < query.length; i++) {
        for (int j = 0; j < query.dim; j++) {
            file << query.data[i * query.dim + j];
            if (j < query.dim - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
}

// 加载查询范围文件
Query loadQueryRangeFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int length = 0, dim = 0;
    std::string line;

    // 读取第一行：长度 维度
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> length >> dim)) {
            throw std::runtime_error("File format error: invalid metadata format");
        }

        if (length <= 0 || dim <= 0) {
            throw std::runtime_error("File format error: length and dimension must be positive");
        }
    } else {
        throw std::runtime_error("File format error: empty file");
    }

    Query query(length, dim, QueryType::RANGE);

    int row = 0;
    while (std::getline(file, line) && row < length) {
        std::istringstream iss(line);
        double value;
        int col = 0;
        int expectedCols = 2 * dim;  // 每行应有 2×维度 个数据

        while (iss >> value && col < expectedCols) {
            query.data[row * expectedCols + col] = value;
            col++;
        }

        if (col != expectedCols) {
            throw std::runtime_error("File format error: expected " +
                                   std::to_string(expectedCols) +
                                   " values but got " + std::to_string(col) +
                                   " at row " + std::to_string(row + 1));
        }
        row++;
    }

    if (row != length) {
        throw std::runtime_error("File format error: row count mismatch");
    }

    file.close();
    return query;
}

// 保存查询范围到文件
void saveQueryRangeToFile(const Query& query, const std::string& filename) {
    if (query.type != QueryType::RANGE) {
        throw std::runtime_error("Can only save range queries with this function");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to create file: " + filename);
    }

    // 第一行：长度 维度
    file << query.length << " " << query.dim << "\n";

    // 数据部分
    for (int i = 0; i < query.length; i++) {
        for (int j = 0; j < 2 * query.dim; j++) {
            file << query.data[i * 2 * query.dim + j];
            if (j < 2 * query.dim - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
}
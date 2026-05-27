//
// Created by cuda01 on 2026/1/12.
//

#include "QueryTest.cuh"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <raft/core/device_mdarray.hpp>
#include "src/func.hpp"

// 提取数据集名称（如 "gist", "sift"）
std::string extractDatasetName(const std::string& path) {
    size_t start = path.find_last_of('/') + 1;
    size_t end = path.find('_', start);
    return path.substr(start, end - start);
}

// 提取范围百分比（如 10, 30, 50...）
int extractRangePercentage(const std::string& path) {
    size_t pos = path.find_last_of('_') + 1;
    size_t dotPos = path.find('.', pos);
    return std::stoi(path.substr(pos, dotPos - pos));
}

std::string extractRangeInfo(const std::string& path) {
    size_t pos = path.find_last_of('_') + 1;
    size_t dotPos = path.find('.', pos);
    std::string rangeStr = path.substr(pos, dotPos - pos);

    size_t dashPos = rangeStr.find('-');
    if (dashPos != std::string::npos) {
        int percentage = std::stoi(rangeStr.substr(0, dashPos));
        int dimensions = std::stoi(rangeStr.substr(dashPos + 1));

        // 计算实际覆盖率
        double actualCoverage = std::pow(percentage / 100.0, dimensions) * 100.0;

        std::ostringstream oss;
        oss << percentage << "%^" << dimensions << "="
            << std::fixed << std::setprecision(2) << actualCoverage << "%";
        return oss.str();
    }

    // 如果没有找到'-'，返回原格式
    return rangeStr + "%";
}

void testQueriesAndSaveResults(const std::string& outputFile, int maxQueryCount,
                              bool loadFromIndex, const std::string& indexPath,
                              const std::string& datasetFile,
                              const std::string& queryPointFile,
                              const std::vector<std::string>& queryRangeFiles) {

    // K值用于KNN查询
    std::vector<size_t> kValues = {3, 5, 50, 100};

    bool isOutputFileExists = std::filesystem::exists(outputFile);
    std::ofstream outFile(outputFile, isOutputFileExists ? std::ios::app : std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFile << std::endl;
        return;
    }

    // 表头
    if (!isOutputFileExists) {
        outFile << std::left
            << "Dataset,"
            << "Size,"
            << "Dim,"
            << "Query Type,"
            << "Query Parameter,"
            << "Query Count,"
            << "Total Time (ms),"
            << "Avg Time (ms),"
            << "Median Log Volume,"
            << "Avg Range Volume,"
            << "Avg Fine Mesh"
            << std::endl;
    }

    // 创建或加载索引
    std::string datasetPath = datasetFile;
    std::string datasetName = extractDatasetName(datasetFile);

    std::unique_ptr<QueryHandler> handler;

    if (loadFromIndex) {
        // 从已有索引加载
        std::cout << "Loading index from: " << indexPath << std::endl;
        try {
            handler = std::make_unique<QueryHandler>(indexPath, true);
            std::cout << "Successfully loaded index for " << datasetName << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to load index for " << datasetName << ": " << e.what() << std::endl;
            std::cerr << "Building new index from dataset..." << std::endl;
            handler = std::make_unique<QueryHandler>(datasetPath);
            handler->saveIndex(indexPath);
            std::cout << "Index saved to: " << indexPath << std::endl;
        }
    }
    else {
        // 从数据集构建新索引
        std::cout << "Building index from dataset: " << datasetPath << std::endl;
        handler = std::make_unique<QueryHandler>(datasetPath);

        // 保存索引以便后续使用
        handler->saveIndex(indexPath);
        std::cout << "Index saved to: " << indexPath << std::endl;
    }

    const auto datasetSize = handler->getSize();
    const auto datasetDim = handler->getDim();

    // 运行POINT查询（KNN）
    std::string queryPath = queryPointFile;

    for (size_t k : kValues) {
        QueryResult result = handler->performQueryWithPreLoadPvals(
            queryPath,
            QueryType::POINT,
            false,
            maxQueryCount,
            k
        );

        if (result.errorCode != 0) {
            std::cerr << "Query failed with K=" << k << std::endl;
            continue;
        }

        // 转换微秒到毫秒
        double totalTimeMs = result.totalTime / 1000.0;
        double avgTimeMs = (result.queryCount > 0) ? (totalTimeMs / result.queryCount) : 0.0;

        // 计算中位数对数体积和平均查询范围体积
        std::string medianLogVolumeStr = "N/A";
        std::string avgVolumeStr = "N/A";
        if (!result.queryRangeVolume.empty()) {
            std::vector<double> sortedVolumes = result.queryRangeVolume;
            std::sort(sortedVolumes.begin(), sortedVolumes.end());

            double medianLogVolume;
            size_t n = sortedVolumes.size();
            if (n % 2 == 0) {
                medianLogVolume = (sortedVolumes[n/2 - 1] + sortedVolumes[n/2]) / 2.0;
            } else {
                medianLogVolume = sortedVolumes[n/2];
            }

            // 输出对数值
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << medianLogVolume;
            medianLogVolumeStr = oss.str();

            // 输出格式化的体积值
            avgVolumeStr = formatLogVolume(medianLogVolume);
        }
        std::string avgFineMeshStr = "N/A";
        if (!result.fineMeshSize.empty()) {
            double avgFineMesh = 0.0;
            for (size_t size : result.fineMeshSize) {
                avgFineMesh += size;
            }
            avgFineMesh /= result.fineMeshSize.size();

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << avgFineMesh;
            avgFineMeshStr = oss.str();
        }

        outFile << std::left
            << datasetName << ","
            << datasetSize << ","
            << datasetDim << ","
            << "POINT (KNN)" << ","
            << ("K=" + std::to_string(k)) << ","
            << result.queryCount << ","
            << std::fixed << std::setprecision(3) << totalTimeMs << ","
            << std::fixed << std::setprecision(6) << avgTimeMs << ","
            << medianLogVolumeStr << ","
            << avgVolumeStr << ","
            << avgFineMeshStr
            << std::endl;
    }

    // 运行RANGE查询
    for (const auto& queryRangeFile : queryRangeFiles) {
        std::string queryPath = queryRangeFile;
        auto rangePercentage = extractRangeInfo(queryRangeFile);

        QueryResult result = handler->performQueryWithPreLoadPvals(
            queryPath,
            QueryType::RANGE,
            false,
            maxQueryCount
        );

        if (result.errorCode != 0) {
            std::cerr << "Query failed with range=" << rangePercentage << std::endl;
            continue;
        }

        // 转换微秒到毫秒
        double totalTimeMs = result.totalTime / 1000.0;
        double avgTimeMs = (result.queryCount > 0) ? (totalTimeMs / result.queryCount) : 0.0;

        // 计算中位数对数体积和平均查询范围体积
        std::string medianLogVolumeStr = "N/A";
        std::string avgVolumeStr = "N/A";
        if (!result.queryRangeVolume.empty()) {
            std::vector<double> sortedVolumes = result.queryRangeVolume;
            std::sort(sortedVolumes.begin(), sortedVolumes.end());

            double medianLogVolume;
            size_t n = sortedVolumes.size();
            if (n % 2 == 0) {
                medianLogVolume = (sortedVolumes[n/2 - 1] + sortedVolumes[n/2]) / 2.0;
            } else {
                medianLogVolume = sortedVolumes[n/2];
            }

            // 输出对数值
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << medianLogVolume;
            medianLogVolumeStr = oss.str();

            // 输出格式化的体积值
            avgVolumeStr = formatLogVolume(medianLogVolume);
        }
        std::string avgFineMeshStr = "N/A";
        if (!result.fineMeshSize.empty()) {
            double avgFineMesh = 0.0;
            for (size_t size : result.fineMeshSize) {
                avgFineMesh += size;
            }
            avgFineMesh /= result.fineMeshSize.size();

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << avgFineMesh;
            avgFineMeshStr = oss.str();
        }

        outFile << std::left
                << datasetName << ","
                << datasetSize << ","
                << datasetDim << ","
                << "RANGE" << ","
                << rangePercentage << ","
                << result.queryCount << ","
                << std::fixed << std::setprecision(3) << totalTimeMs << ","
                << std::fixed << std::setprecision(6) << avgTimeMs << ","
                << medianLogVolumeStr << ","
                << avgVolumeStr << ","
                << avgFineMeshStr
                << std::endl;
    }

    outFile.close();

    std::cout << "Results saved to: " << outputFile << std::endl;
}
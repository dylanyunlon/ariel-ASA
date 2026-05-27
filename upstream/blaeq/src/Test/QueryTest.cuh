//
// Created by cuda01 on 2026/1/12.
//

#ifndef BLAEQ_CUDA_QUERYTEST_CUH
#define BLAEQ_CUDA_QUERYTEST_CUH

#include "src/Query/Query.cuh"

void testQueriesAndSaveResults(const std::string& outputFile, int maxQueryCount,
                              bool loadFromIndex, const std::string& indexPath,
                              const std::string& datasetFile,
                              const std::string& queryPointFile,
                              const std::vector<std::string>& queryRangeFiles);

#endif //BLAEQ_CUDA_QUERYTEST_CUH
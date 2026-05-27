//
// Created by Mgepahmge on 2025/12/12.
//

#ifndef CUDADB_KNN_CUH
#define CUDADB_KNN_CUH

#include "src/Data_Structures/Data_Structures.cuh"

bool* knnPruning(const size_t k, const size_t p, const size_t dim, const size_t length, const double* query_point,
                 const double* centroids, const double* radius,
                 const size_t* cluster_sizes, const size_t* indexs, size_t& out_selected_count);

#endif //CUDADB_KNN_CUH
//
// Created by Mgepahmge on 2025/12/18.
//

#ifndef CUDADB_RANGEPRUNING_CUH
#define CUDADB_RANGEPRUNING_CUH

bool* rangePruning(const double* lowBounds, const double* upBounds, const size_t dim, const double* centroids,
                   const double* radius, const size_t p, const size_t* indexs, const size_t length, size_t& out_selected_count);

#endif //CUDADB_RANGEPRUNING_CUH
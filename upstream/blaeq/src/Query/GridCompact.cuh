//
// Created by Mgepahmge on 2025/12/17.
//

#ifndef CUDADB_GRIDCOMPACT_CUH
#define CUDADB_GRIDCOMPACT_CUH
#include "src/Data_Structures/Data_Structures.cuh"

GridAsSparseMatrix* compactGrid(const GridAsSparseMatrix& grid, const bool* mask, const size_t validCount);



#endif //CUDADB_GRIDCOMPACT_CUH
//
// Created by cuda01 on 2026/1/5.
//

#ifndef BLAEQ_CUDA_SETUP_CUH
#define BLAEQ_CUDA_SETUP_CUH
#include "src/Data_Structures/File.cuh"
#include "src/Kmeans/CUDAKmeans.cuh"
#include "src/Query/check.cuh"


size_t Compute_Layer_nums(size_t N);
size_t Compute_Centroid_nums(size_t data_nums, size_t ratio);
SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t P_row_len, size_t P_col_len, CUDAKmeans* KMeans_Ptr,
                                             size_t*& map);
double* Compute_Max_Radius(size_t D, const size_t* centroid_col_res, const size_t* sort_to_original_reflections,
                           CUDAKmeans* KMeans_Ptr);
void deleteGrid(GridAsSparseMatrix* grid);
std::string getQueryTypeString(QueryType qType);

template <typename T>
void freeMemoryDependAttr(T* point, cudaPointerAttributes attr) {
    if (attr.type == cudaMemoryTypeDevice) {
        CUDA_CHECK(cudaFree(point));
    }
    else if (attr.type == cudaMemoryTypeHost) {
        delete[] point;
    }
    else if (attr.type == cudaMemoryTypeManaged) {
        CUDA_CHECK(cudaFree(point));
    }
    else {
        delete[] point;
    }
}

#endif //BLAEQ_CUDA_SETUP_CUH
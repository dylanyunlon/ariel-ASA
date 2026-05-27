//
// Created by cuda01 on 2026/1/5.
//

#include "Setup.cuh"
#include "src/func.hpp"

size_t Compute_Layer_nums(size_t N) {
    return 4;
}

size_t Compute_Centroid_nums(size_t data_nums, size_t ratio) {
    if (data_nums / ratio == 0) return 1;
    return data_nums / ratio;
}

SparseTensorCscFormat* Genenate_One_P_Tensor(size_t D, size_t P_row_len, size_t P_col_len, CUDAKmeans* KMeans_Ptr,
                                             size_t*& map) {
    assert(P_row_len > 0);
    assert(P_col_len > 0);
    // 1. get original data( Coreast Mesh, Fine Mesh, Labels)
    auto& Coreast_Original_Mesh = KMeans_Ptr->getCentroids();
    auto& Fine_Original_Mesh = KMeans_Ptr->getdatas();
    auto& labels = KMeans_Ptr->getLabels();
    // 2. get P nnz_per_col array
    std::vector<size_t> labels_mcv_list(P_col_len, 0);
    std::for_each(labels.begin(), labels.end(),
                  [&labels_mcv_list](int id) { ++labels_mcv_list[id]; });

    size_t max_nnz_per_col = 0;
    max_nnz_per_col = *std::max_element(labels_mcv_list.begin(), labels_mcv_list.end());
    std::cout << "max_nnz_per_col are " << max_nnz_per_col << std::endl;
    assert(max_nnz_per_col > 0);
    // xak note :: why + 10 ? To prevent the array boundaries problem
    double* P_vals_batch = new double[D * (max_nnz_per_col + 10)]();

    // 3. new P_Sparse_Tensor_Csc_Format
    SparseTensorCscFormat* P_Sparse_Tensor_Csc_Format = new SparseTensorCscFormat(
        D, P_row_len, P_col_len, labels_mcv_list);

    // 4. do sort and copy
    std::vector<size_t> sort_indices_vec = Sort::Sorted_Layer_With_Original_idxs(labels);
    map = new size_t[sort_indices_vec.size()];
    std::copy(sort_indices_vec.begin(), sort_indices_vec.end(), map);

    // 5. do-loop insert batchs
    const size_t* P_col_res = P_Sparse_Tensor_Csc_Format->get_col_res();
    for (size_t i = 0; i < P_col_len; i++) {
        size_t begin_pos = P_col_res[i];
        size_t end_pos = P_col_res[i + 1];
        assert(begin_pos <= end_pos);
        auto& centroid_val = Coreast_Original_Mesh[i];
        size_t writeIdx = 0;
        for (size_t j = begin_pos; j < end_pos; j++) {
            size_t original_id = sort_indices_vec[j];
            auto& original_val = Fine_Original_Mesh[original_id];
            for (size_t k = 0; k < D; k++) {
                if (Comp::isZero(centroid_val[k])) {
                    std::cout << "There is num / 0 serious error" << std::endl;
                    assert(false);
                }
                P_vals_batch[writeIdx * D + k] = original_val[k] / centroid_val[k];
            }
            writeIdx++;
        }
        P_Sparse_Tensor_Csc_Format->Insert_One_Batch(P_vals_batch, begin_pos, end_pos);
    }

    // 6. debug
    // nothing !!!

    // 7. ret
    delete[] P_vals_batch;
    return P_Sparse_Tensor_Csc_Format;
}


double* Compute_Max_Radius(size_t D, const size_t* centroid_col_res, const size_t* sort_to_original_reflections,
                           CUDAKmeans* KMeans_Ptr) {
    auto& Coreast_Original_Mesh = KMeans_Ptr->getCentroids();
    auto& Fine_Original_Mesh = KMeans_Ptr->getdatas();
    size_t Centroids_nums = Coreast_Original_Mesh.size();
    double* max_radius = new double[Centroids_nums];
    for (size_t i = 0; i < Centroids_nums; i++) {
        size_t begin_pos = centroid_col_res[i];
        size_t end_pos = centroid_col_res[i + 1];
        auto& coreast_mesh_val = Coreast_Original_Mesh[i];
        double max_distance = 0.0;
        for (size_t sort_idx = begin_pos; sort_idx < end_pos; sort_idx++) {
            size_t abs_idx = sort_to_original_reflections[sort_idx];
            auto& fine_mesh_val = Fine_Original_Mesh[abs_idx];
            max_distance = std::max(max_distance, dist::euclidean(coreast_mesh_val, fine_mesh_val));
        }
        // assert(!Comp::isZero(max_distance));
        max_radius[i] = max_distance;
    }
    return max_radius;
}



void deleteGrid(GridAsSparseMatrix* grid) {
    cudaPointerAttributes attr{};
    CUDA_CHECK(cudaPointerGetAttributes(&attr, grid->get_ids_()));
    freeMemoryDependAttr(grid->get_ids_(), attr);
    CUDA_CHECK(cudaPointerGetAttributes(&attr, grid->get_vals_()));
    freeMemoryDependAttr(grid->get_vals_(), attr);
    grid->set_ids(nullptr);
    grid->set_vals(nullptr);
    delete grid;
}

std::string getQueryTypeString(const QueryType qType) {
    switch (qType) {
    case QueryType::RANGE:
        return "RANGE";
    case QueryType::POINT:
        return "POINT";
    default:
        return "UNKNOWN";
    }
}
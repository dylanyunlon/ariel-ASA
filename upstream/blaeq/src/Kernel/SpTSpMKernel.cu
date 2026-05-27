#include "src/Kernel/SpTSpMKernel.cuh"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <assert.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <cassert>
#include <cstring>

constexpr int UNROLL_FACTOR = 4;

namespace kernel {
// 声明
    void SpTSpMCPUKernel(const size_t curr_mesh_nnz_nums, const size_t D, 
                         const std::vector<std::vector<double>>& h_P_vals,const std::vector<size_t>& h_P_col_res, 
                         const std::vector<std::vector<double>>& h_curr_M_vals, const std::vector<size_t>& h_curr_M_nnz_ids,
                         std::vector<std::vector<double>>& h_next_M_vals, std::vector<size_t>& h_next_M_ids)
    {
        size_t idx = 0;
        for(size_t i = 0; i < curr_mesh_nnz_nums; i++){
            size_t P_Tensor_col_idx = h_curr_M_nnz_ids[i];
            auto& curr_mesh_nnz_val = h_curr_M_vals[P_Tensor_col_idx];
            size_t begin_pos = h_P_col_res[P_Tensor_col_idx];
            size_t end_pos = h_P_col_res[P_Tensor_col_idx+1];
            assert(begin_pos <= end_pos);
            for(size_t j = begin_pos; j < end_pos; j++){
                h_next_M_ids[idx] = j;
                for(size_t k = 0; k < D; k++){
                    h_next_M_vals[idx][k] = h_P_vals[j][k] * curr_mesh_nnz_val[k];
                }
                idx++;
            }
        }
    }

    void SpTSpMGPUKernelAOS(const size_t next_mesh_nnz_nums, const size_t D,
                            const double* d_P_vals, const double* d_curr_M_vals, 
                            const size_t* d_next_M_nnz_vals_P_row_ids, const size_t* d_next_M_nnz_vals_P_col_ids,
                            double* d_next_M_vals)
    { 
        std:: cout << "prepare to do tensor multiply multigrid in gpu aos version" << std::endl;
        size_t next_mesh_nnz_dim_nums = next_mesh_nnz_nums * D;
        // 1. gpu acc 
        dim3 block(512);
        dim3 grid((next_mesh_nnz_dim_nums + UNROLL_FACTOR * block.x - 1) / (UNROLL_FACTOR * block.x));
        SpTSpMKernelAOSImpl<<<grid, block>>>(next_mesh_nnz_nums, D, d_P_vals, d_curr_M_vals, d_next_M_nnz_vals_P_row_ids, d_next_M_nnz_vals_P_col_ids, d_next_M_vals);
        cudaDeviceSynchronize();

        std::cout << "finish the kernel function " << std::endl;
    }


    // kernel func - for aos 
    __global__ void SpTSpMKernelAOSImpl(const unsigned int next_mesh_nnz_nums, const size_t D,                            
                                        const double* d_P_vals, const double* d_curr_M_vals,     
                                        const size_t* d_next_M_nnz_vals_P_row_ids, const size_t* d_next_M_nnz_vals_P_col_ids,
                                        double* d_next_M_vals) 
    {
        const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        const auto stride = blockDim.x * gridDim.x; 
    #pragma unroll 
        for (auto i = 0; i < UNROLL_FACTOR; ++i) {
            if (const auto index = idx + i * stride; index < next_mesh_nnz_nums * D){
                const auto col_idx = d_next_M_nnz_vals_P_col_ids[index / D];
                const auto row_idx = d_next_M_nnz_vals_P_row_ids[index / D];
                d_next_M_vals[index] = d_P_vals[row_idx * D + index % D] * d_curr_M_vals[col_idx * D + index % D];
            }
        }
    }       

}
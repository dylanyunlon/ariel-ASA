#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace kernel {
// 声明
    void SpTSpMCPUKernel(
        const size_t curr_mesh_nnz_nums,
        const size_t D,
        const std::vector<std::vector<double>>& h_P_vals,
        const std::vector<size_t>& h_P_col_res,
        const std::vector<std::vector<double>>& h_curr_M_vals,
        const std::vector<size_t>& h_curr_M_nnz_ids,
        std::vector<std::vector<double>>& h_next_M_vals,
        std::vector<size_t>& h_next_M_ids);

    void SpTSpMGPUKernelAOS(
        const size_t next_mesh_nnz_nums, 
        const size_t D,              
        const double* d_P_vals, 
        const double* d_curr_M_vals,     
        const size_t* d_next_M_nnz_vals_P_row_ids, 
        const size_t* d_next_M_nnz_vals_P_col_ids,
        double* d_next_M_vals);

    __global__ void SpTSpMKernelAOSImpl(
        const unsigned int next_mesh_nnz_nums, 
        const size_t D,                            
        const double* d_P_vals, 
        const double* d_curr_M_vals,     
        const size_t* d_next_M_nnz_vals_P_row_ids, 
        const size_t* d_next_M_nnz_vals_P_col_ids,
        double* d_next_M_vals); 

    void SpTSpMGPUKernelSOA();

    __global__ void SpTSpMKernelSOAImpl();

    void SpTSpMGPUKernelIntuition();

    __global__ void SpTSpMKernelIntuitionImpl();

}
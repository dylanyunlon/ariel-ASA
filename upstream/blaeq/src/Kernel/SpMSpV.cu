//
// Created by Mgepahmge on 2025/12/19.
//
#include "SpMSpV.cuh"
#include "src/Query/check.cuh"
#include "src/utils/NVTXProfiler.cuh"



/**
 * @brief Multiply a sparse tensor in CSC format with a grid represented as a sparse matrix.
 *
 * @param[in] P Pointer to the sparse tensor in CSC format.(device/host)
 * @param[in] grid Pointer to the grid represented as a sparse matrix.(device)
 *
 * @return GridAsSparseMatrix The resulting grid after multiplication.(device)
 */
GridAsSparseMatrix* SpTSpMMultiplication(SparseTensorCscFormat* P, GridAsSparseMatrix* grid) {
    NvtxProfiler profiler("SpTSpMMultiplication", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);
    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto P_col_nums = P->get_col_nums();
    const auto grid_size = grid->get_nnz_nums();

    size_t* h_matrixColPtr = nullptr;
    size_t* h_matrixRowInd = nullptr;
    double* h_matrixData = nullptr;
    auto* h_vectorIndex = new size_t[grid_size];
    auto* h_vectorData = new double[grid_size * numDims];
    auto* d_vectorIndex = const_cast<size_t*>(grid->get_ids_());
    auto* d_vectorData = const_cast<double*>(grid->get_vals_());
    CUDA_CHECK(cudaMemcpy(h_vectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vectorData, grid->get_vals_(), grid_size * numDims * sizeof(double), cudaMemcpyDeviceToHost));

    cudaPointerAttributes P_attrib{};
    CUDA_CHECK(cudaPointerGetAttributes(&P_attrib, P->get_vals()));
    if (P_attrib.type == cudaMemoryTypeDevice) {
        h_matrixColPtr = new size_t[P_col_nums + 1];
        CUDA_CHECK(cudaMemcpy(h_matrixColPtr, P->get_col_res(), (P_col_nums + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));
        const auto nnz_size = h_matrixColPtr[P_col_nums];
        h_matrixRowInd = new size_t[nnz_size];
        h_matrixData = new double[nnz_size * numDims];
        CUDA_CHECK(cudaMemcpy(h_matrixRowInd, P->get_row_ids(), nnz_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_matrixData, P->get_vals(), nnz_size * numDims * sizeof(double), cudaMemcpyDeviceToHost));
    } else
    {
        h_matrixColPtr = const_cast<size_t*>(P->get_col_res());
        h_matrixRowInd = const_cast<size_t*>(P->get_row_ids());
        h_matrixData = const_cast<double*>(P->get_vals());
    }

    unsigned int numProcessedNonZero = 0;
    // Step 1
    // 计算稀疏矩阵中待处理的非零元数量
    NvtxProfiler innerProfiler1("计算待处理非零元数量", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Rose);
    for (auto i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        numProcessedNonZero += (h_matrixColPtr[col + 1] - h_matrixColPtr[col]);
    }
    innerProfiler1.release();

    // Step 2
    // 分配中间数据存储空间
    NvtxProfiler innerProfiler2("分配中间数据存储空间", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    auto* h_processedColInd = new size_t[numProcessedNonZero];
    auto* h_processedValues = new double[numProcessedNonZero * numDims];
    auto* h_processedRowInd = new size_t[numProcessedNonZero];
    size_t* d_processedColInd = nullptr;
    double* d_processedValues = nullptr;
    size_t* d_processedRowInd = nullptr;
    CUDA_CHECK(cudaMalloc(&d_processedColInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedValues, numProcessedNonZero * numDims * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_processedRowInd, numProcessedNonZero * sizeof(size_t)));
    innerProfiler2.release();

    // Step 3
    // 提取有效数据
    NvtxProfiler innerProfiler3("提取有效数据", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Magenta);
    unsigned int writeIdx = 0;
    for (auto i = 0; i < grid_size; ++i) {
        auto col = h_vectorIndex[i];
        auto colStart = h_matrixColPtr[col];
        auto colEnd = h_matrixColPtr[col + 1];

        for (auto j = colStart; j < colEnd; ++j) {
            h_processedColInd[writeIdx] = i;
            h_processedRowInd[writeIdx] = h_matrixRowInd[j];
            // 复制对应维度的值
            for (auto d = 0; d < numDims; ++d) {
                h_processedValues[writeIdx * numDims + d] =
                    h_matrixData[j * numDims + d];
            }
            writeIdx++;
        }
    }
    innerProfiler3.release();

    // 将处理后的数据拷贝到设备端
    CUDA_CHECK(cudaMemcpy(d_processedColInd, h_processedColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedRowInd, h_processedRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedValues, h_processedValues, numProcessedNonZero * numDims * sizeof(double), cudaMemcpyHostToDevice));

    size_t* yIndex;
    double* yValue;
    const auto totalNumNoneZero = numProcessedNonZero * numDims;
    CUDA_CHECK(cudaMalloc(&yValue, totalNumNoneZero * sizeof(double)));
    dim3 blockSize(512);
    dim3 gridSize((totalNumNoneZero + UNROLL_FACTOR * blockSize.x - 1) / (UNROLL_FACTOR * blockSize.x));
    SpMSpVKernelAOS<<<gridSize, blockSize>>>(yValue, d_processedColInd, d_processedValues, d_vectorData,
                                                        numDims, totalNumNoneZero);
    yIndex = d_processedRowInd;

    // Step 4
    // 在kernel运行期间并发的清理资源

    if (P_attrib.type == cudaMemoryTypeDevice) {
        delete[] h_matrixColPtr;
        delete[] h_matrixRowInd;
        delete[] h_matrixData;
    }
    delete[] h_vectorIndex;
    delete[] h_vectorData;
    delete[] h_processedColInd;
    delete[] h_processedValues;
    delete[] h_processedRowInd;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_processedColInd));
    CUDA_CHECK(cudaFree(d_processedValues));

    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, yIndex, yValue};
}

/**
 * @brief V2版本：优化版本，避免CPU端大量数据复制
 *
 * 主要改进：
 * 1. 不再复制矩阵values数据到新数组
 * 2. 存储原始矩阵中的位置索引
 * 3. 在kernel中直接访问原始矩阵数据
 */
GridAsSparseMatrix* SpTSpMMultiplication_v2(SparseTensorCscFormat* P, GridAsSparseMatrix* grid) {
    NvtxProfiler profiler("SpTSpMMultiplication_v2", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);
    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto P_col_nums = P->get_col_nums();
    const auto grid_size = grid->get_nnz_nums();

    size_t* h_matrixColPtr = nullptr;
    size_t* h_matrixRowInd = nullptr;
    double* d_matrixData = nullptr;
    auto* h_vectorIndex = new size_t[grid_size];
    auto* d_vectorData = const_cast<double*>(grid->get_vals_());
    CUDA_CHECK(cudaMemcpy(h_vectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    cudaPointerAttributes P_attrib{};
    CUDA_CHECK(cudaPointerGetAttributes(&P_attrib, P->get_vals()));

    if (P_attrib.type == cudaMemoryTypeDevice) {
        h_matrixColPtr = new size_t[P_col_nums + 1];
        CUDA_CHECK(cudaMemcpy(h_matrixColPtr, P->get_col_res(), (P_col_nums + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));
        const auto nnz_size = h_matrixColPtr[P_col_nums];
        h_matrixRowInd = new size_t[nnz_size];
        CUDA_CHECK(cudaMemcpy(h_matrixRowInd, P->get_row_ids(), nnz_size * sizeof(size_t), cudaMemcpyDeviceToHost));

        // 直接使用设备端的原始数据指针
        d_matrixData = const_cast<double*>(P->get_vals());
    } else {
        h_matrixColPtr = const_cast<size_t*>(P->get_col_res());
        h_matrixRowInd = const_cast<size_t*>(P->get_row_ids());

        // 主机端数据需要拷贝到设备端
        const auto nnz_size = h_matrixColPtr[P_col_nums];
        CUDA_CHECK(cudaMalloc(&d_matrixData, nnz_size * numDims * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_matrixData, P->get_vals(), nnz_size * numDims * sizeof(double), cudaMemcpyHostToDevice));
    }

    unsigned int numProcessedNonZero = 0;
    // Step 1: 计算待处理的非零元数量
    NvtxProfiler innerProfiler1("计算待处理非零元数量", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Rose);
    for (auto i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        numProcessedNonZero += (h_matrixColPtr[col + 1] - h_matrixColPtr[col]);
    }
    innerProfiler1.release();

    // Step 2: 分配索引数据存储空间
    NvtxProfiler innerProfiler2("分配索引数据存储空间", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    auto* h_processedColInd = new size_t[numProcessedNonZero];
    auto* h_processedRowInd = new size_t[numProcessedNonZero];
    auto* h_processedMatrixPos = new size_t[numProcessedNonZero]; // 存储原始矩阵位置
    size_t* d_processedColInd = nullptr;
    size_t* d_processedRowInd = nullptr;
    size_t* d_processedMatrixPos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_processedColInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedRowInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedMatrixPos, numProcessedNonZero * sizeof(size_t)));
    innerProfiler2.release();

    // Step 3: 提取索引数据（不复制values）
    NvtxProfiler innerProfiler3("提取索引数据", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Magenta);
    unsigned int writeIdx = 0;
    for (auto i = 0; i < grid_size; ++i) {
        auto col = h_vectorIndex[i];
        auto colStart = h_matrixColPtr[col];
        auto colEnd = h_matrixColPtr[col + 1];

        for (auto j = colStart; j < colEnd; ++j) {
            h_processedColInd[writeIdx] = i;
            h_processedRowInd[writeIdx] = h_matrixRowInd[j];
            h_processedMatrixPos[writeIdx] = j; // 记录原始矩阵中的位置
            writeIdx++;
        }
    }
    innerProfiler3.release();

    // 将索引数据拷贝到设备端
    CUDA_CHECK(cudaMemcpy(d_processedColInd, h_processedColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedRowInd, h_processedRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedMatrixPos, h_processedMatrixPos, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* yIndex;
    double* yValue;
    const auto totalNumNoneZero = numProcessedNonZero * numDims;
    CUDA_CHECK(cudaMalloc(&yValue, totalNumNoneZero * sizeof(double)));

    dim3 blockSize(512);
    dim3 gridSize_kernel((totalNumNoneZero + UNROLL_FACTOR * blockSize.x - 1) / (UNROLL_FACTOR * blockSize.x));

    // 调用v2 kernel
    SpMSpVKernelAOS_v2<<<gridSize_kernel, blockSize>>>(yValue, d_processedColInd, d_processedMatrixPos,
                                                        d_matrixData, d_vectorData,
                                                        numDims, totalNumNoneZero);
    yIndex = d_processedRowInd;

    // Step 4: 清理资源
    if (P_attrib.type == cudaMemoryTypeDevice) {
        delete[] h_matrixColPtr;
        delete[] h_matrixRowInd;
    } else {
        CUDA_CHECK(cudaFree(d_matrixData));
    }
    delete[] h_vectorIndex;
    delete[] h_processedColInd;
    delete[] h_processedRowInd;
    delete[] h_processedMatrixPos;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_processedColInd));
    CUDA_CHECK(cudaFree(d_processedMatrixPos));

    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, yIndex, yValue};
}

/**
 * @brief V3版本：预加载数据到GPU
 *
 * 主要改进：
 * 1. 预先将P举证的values加载到GPU，避免大量内存拷贝
 */
GridAsSparseMatrix* SpTSpMMultiplication_v3(SparseTensorCscFormat* P, GridAsSparseMatrix* grid, double* d_P_values) {
    NvtxProfiler profiler("SpTSpMMultiplication_v3", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Orange);
    const auto numDims = grid->get_dimensions();
    const auto P_row_nums = P->get_row_nums();
    const auto P_col_nums = P->get_col_nums();
    const auto grid_size = grid->get_nnz_nums();

    size_t* h_matrixColPtr = nullptr;
    size_t* h_matrixRowInd = nullptr;
    double* d_matrixData = d_P_values;
    auto* h_vectorIndex = new size_t[grid_size];
    auto* d_vectorData = const_cast<double*>(grid->get_vals_());
    CUDA_CHECK(cudaMemcpy(h_vectorIndex, grid->get_ids_(), grid_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    cudaPointerAttributes P_attrib{};
    CUDA_CHECK(cudaPointerGetAttributes(&P_attrib, P->get_vals()));

    if (P_attrib.type == cudaMemoryTypeDevice) {
        h_matrixColPtr = new size_t[P_col_nums + 1];
        CUDA_CHECK(cudaMemcpy(h_matrixColPtr, P->get_col_res(), (P_col_nums + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));
        const auto nnz_size = h_matrixColPtr[P_col_nums];
        h_matrixRowInd = new size_t[nnz_size];
        CUDA_CHECK(cudaMemcpy(h_matrixRowInd, P->get_row_ids(), nnz_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    } else {
        h_matrixColPtr = const_cast<size_t*>(P->get_col_res());
        h_matrixRowInd = const_cast<size_t*>(P->get_row_ids());
    }

    unsigned int numProcessedNonZero = 0;
    // Step 1: 计算待处理的非零元数量
    NvtxProfiler innerProfiler1("计算待处理非零元数量", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Rose);
    for (auto i = 0; i < grid_size; ++i) {
        const auto col = h_vectorIndex[i];
        numProcessedNonZero += (h_matrixColPtr[col + 1] - h_matrixColPtr[col]);
    }
    innerProfiler1.release();

    // Step 2: 分配索引数据存储空间
    NvtxProfiler innerProfiler2("分配索引数据存储空间", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::LimeGreen);
    auto* h_processedColInd = new size_t[numProcessedNonZero];
    auto* h_processedRowInd = new size_t[numProcessedNonZero];
    auto* h_processedMatrixPos = new size_t[numProcessedNonZero]; // 存储原始矩阵位置
    size_t* d_processedColInd = nullptr;
    size_t* d_processedRowInd = nullptr;
    size_t* d_processedMatrixPos = nullptr;
    CUDA_CHECK(cudaMalloc(&d_processedColInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedRowInd, numProcessedNonZero * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_processedMatrixPos, numProcessedNonZero * sizeof(size_t)));
    innerProfiler2.release();

    // Step 3: 提取索引数据（不复制values）
    NvtxProfiler innerProfiler3("提取索引数据", NvtxProfiler::ColorMode::Fixed, NvtxProfilerColor::Magenta);
    unsigned int writeIdx = 0;
    for (auto i = 0; i < grid_size; ++i) {
        auto col = h_vectorIndex[i];
        auto colStart = h_matrixColPtr[col];
        auto colEnd = h_matrixColPtr[col + 1];

        for (auto j = colStart; j < colEnd; ++j) {
            h_processedColInd[writeIdx] = i;
            h_processedRowInd[writeIdx] = h_matrixRowInd[j];
            h_processedMatrixPos[writeIdx] = j; // 记录原始矩阵中的位置
            writeIdx++;
        }
    }
    innerProfiler3.release();

    // 将索引数据拷贝到设备端
    CUDA_CHECK(cudaMemcpy(d_processedColInd, h_processedColInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedRowInd, h_processedRowInd, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_processedMatrixPos, h_processedMatrixPos, numProcessedNonZero * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t* yIndex;
    double* yValue;
    const auto totalNumNoneZero = numProcessedNonZero * numDims;
    CUDA_CHECK(cudaMalloc(&yValue, totalNumNoneZero * sizeof(double)));

    dim3 blockSize(512);
    dim3 gridSize_kernel((totalNumNoneZero + UNROLL_FACTOR * blockSize.x - 1) / (UNROLL_FACTOR * blockSize.x));

    // 调用v2 kernel
    SpMSpVKernelAOS_v2<<<gridSize_kernel, blockSize>>>(yValue, d_processedColInd, d_processedMatrixPos,
                                                        d_matrixData, d_vectorData,
                                                        numDims, totalNumNoneZero);
    yIndex = d_processedRowInd;

    // Step 4: 清理资源
    if (P_attrib.type == cudaMemoryTypeDevice) {
        delete[] h_matrixColPtr;
        delete[] h_matrixRowInd;
    } else {
    }
    delete[] h_vectorIndex;
    delete[] h_processedColInd;
    delete[] h_processedRowInd;
    delete[] h_processedMatrixPos;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_processedColInd));
    CUDA_CHECK(cudaFree(d_processedMatrixPos));

    return new GridAsSparseMatrix{P_row_nums, numDims, numProcessedNonZero, yIndex, yValue};
}
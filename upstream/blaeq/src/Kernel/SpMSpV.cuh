#ifndef SPMSPV_SPMSPV_CUH
#define SPMSPV_SPMSPV_CUH

#include <cstring>
#include "src/Data_Structures/Data_Structures.cuh"

constexpr unsigned int UNROLL_FACTOR = 4;

// 计算SpMSpV，按照按非零元划分的策略，每个线程处理UNROLL_FACTOR个非零元，并且兼容unroll优化
// 内存布局为AOS
// 支持多维度数据
template <typename Integer, typename Real>
__global__ void SpMSpVKernelAOS(Real* yValue, const Integer* colInd, const Real* value, const Real* xValue,
                                const unsigned int numDims, const unsigned int totalNumNoneZero) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;
#pragma unroll
    for (auto i = 0; i < UNROLL_FACTOR; ++i) {
        if (const auto index = idx + i * stride; index < totalNumNoneZero) {
            const auto col = colInd[index / numDims];
            yValue[index] = value[index] * xValue[col * numDims + index % numDims];
        }
    }
}

// 计算SpMSpV，按照按非零元划分的策略，每个线程处理numDims个非零元
// 内存布局为SOA
template <typename Integer, typename Real>
__global__ void SpMSpVKernelSOA(Real* yValue, const Integer* colInd, const Real* value, const Real* xValue,
                                const unsigned int numDims, const unsigned int numNoneZero,
                                const unsigned int numRowsX) {
    if (const auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numNoneZero) {
        const auto col = colInd[idx];
#pragma unroll
        for (auto d = 0; d < numDims; ++d) {
            const auto index = d * numNoneZero + idx;
            yValue[index] = value[index] * xValue[d * numRowsX + col];
        }
    }
}

// 按照AOS内存布局实现的SpMSpV处理句柄
// 当维度数较多的时候可以使用此版本
template <typename Integer, typename Real>
class SpMSpVHandleAOS {
public:
    using intType = Integer;
    using realType = Real;

    SpMSpVHandleAOS(const unsigned int numDims,
                    const unsigned int numMatrixRows,
                    const unsigned int numMatrixCols,
                    const unsigned int numVectorRows,
                    Integer* matrixColPtr,
                    Integer* matrixRowInd,
                    Real* matrixValues,
                    Integer* vectorIndex,
                    Real* vectorValues)
        : numDims(numDims),
          numMatrixRows(numMatrixRows),
          numMatrixCols(numMatrixCols),
          numVectorRows(numVectorRows),
          matrixColPtr(matrixColPtr),
          matrixRowInd(matrixRowInd),
          matrixValues(matrixValues),
          vectorIndex(vectorIndex),
          vectorValues(vectorValues), vectorValuesDevice(nullptr), processedColInd(nullptr), processedRowInd(nullptr),
          processedValues(nullptr) {
    }

    ~SpMSpVHandleAOS() {
        cudaFree(processedColInd);
        delete[] processedRowInd;
        cudaFree(processedValues);
        cudaFree(vectorValuesDevice);
    }

    // 假设为AOS内存布局，并且所有数据存储在主机端
    // 后续可扩展为设备端存储，降低内存操作开销，进一步优化性能
    // [{x_0, y_0, z_0, ...}, {x_1, y_1, z_1, ...}, ..., {x_n, y_n, z_n, ...}]
    // 预处理矩阵，提取其中的有效数据（与输入稀疏向量对应的列）
    // 预处理后的矩阵存储有效数据，以及存储用于在输入向量中定位的相对索引数据
    void setup() {
        cudaMalloc(&vectorValuesDevice, numDims * numVectorRows * sizeof(Real));
        cudaMemcpy(vectorValuesDevice, vectorValues, numDims * numVectorRows * sizeof(Real), cudaMemcpyHostToDevice);
        // Step 1
        // 计算稀疏矩阵中待处理的非零元数量
        for (auto i = 0; i < numVectorRows; ++i) {
            auto col = vectorIndex[i];
            numProcessedNonZero += matrixColPtr[col + 1] - matrixColPtr[col];
        }

        // Step 2
        // 分配存储空间
        // 此处使用统一内存，后续可根据需要调整
        // 行索引使用主机端内存
        cudaMallocManaged(&processedColInd, numProcessedNonZero * sizeof(Integer));
        cudaMallocManaged(&processedValues, numDims * numProcessedNonZero * sizeof(Real));
        processedRowInd = new Integer[numProcessedNonZero];

        // Step 3
        // 提取有效数据，依然按照AOS布局
        // 此处可尝试使用GPU并行处理，但需要辅助数据来存储每列的起始位置
        unsigned int writeIdx = 0;
        for (auto i = 0; i < numVectorRows; ++i) {
            auto col = vectorIndex[i];
            auto colStart = matrixColPtr[col];
            auto colEnd = matrixColPtr[col + 1];

            for (auto j = colStart; j < colEnd; ++j) {
                processedColInd[writeIdx] = i;
                processedRowInd[writeIdx] = matrixRowInd[j];
                // 复制对应维度的值
                for (auto d = 0; d < numDims; ++d) {
                    processedValues[writeIdx * numDims + d] =
                        matrixValues[j * numDims + d];
                }
                writeIdx++;
            }
        }
    }

    // 计算
    // 假设y也为稀疏向量，并且也为AOS内存布局
    // 假设y尚未分配存储空间
    unsigned int compute(Integer** yIndex, Real** yValue) {
        const auto totalNumNoneZero = numDims * numProcessedNonZero;
        Real* yValueDevice;
        Integer* yIndexDevice;
        cudaMalloc(&yValueDevice, totalNumNoneZero * sizeof(Real));
        cudaMalloc(&yIndexDevice, numProcessedNonZero * sizeof(Integer));
        dim3 block(512);
        dim3 grid((totalNumNoneZero + UNROLL_FACTOR * block.x - 1) / (UNROLL_FACTOR * block.x));
        SpMSpVKernelAOS<Integer, Real><<<grid, block>>>(yValueDevice, processedColInd, processedValues, vectorValuesDevice,
                                                        numDims, totalNumNoneZero);
        cudaDeviceSynchronize();
        cudaMemcpy(*yIndex, processedRowInd, numProcessedNonZero * sizeof(Integer), cudaMemcpyHostToHost);
        yValue = &yValueDevice;
        yIndex = &yIndexDevice;
        return numProcessedNonZero;
    }

private:
    unsigned int numDims;
    unsigned int numMatrixRows;
    unsigned int numMatrixCols;
    unsigned int numVectorRows;
    unsigned int numProcessedNonZero{};
    Integer* matrixColPtr;
    Integer* matrixRowInd;
    Real* matrixValues;
    Integer* vectorIndex;
    Real* vectorValues;
    Real* vectorValuesDevice;
    Integer* processedColInd;
    Integer* processedRowInd;
    Real* processedValues;
};

// 按照SOA内存布局实现的SpMSpV处理句柄
// 当维度数较少的时候可以使用此版本
template <typename Integer, typename Real>
class SpMSpVHandleSOA {
public:
    using intType = Integer;
    using realType = Real;

    SpMSpVHandleSOA(const unsigned int numDims,
                    const unsigned int numMatrixRows,
                    const unsigned int numMatrixCols,
                    const unsigned int numVectorRows,
                    Integer* matrixColPtr,
                    Integer* matrixRowInd,
                    Real* matrixValues,
                    Integer* vectorIndex,
                    Real* vectorValues)
        : numDims(numDims),
          numMatrixRows(numMatrixRows),
          numMatrixCols(numMatrixCols),
          numVectorRows(numVectorRows),
          matrixColPtr(matrixColPtr),
          matrixRowInd(matrixRowInd),
          matrixValues(matrixValues),
          vectorIndex(vectorIndex),
          vectorValues(vectorValues), vectorValuesDevice(nullptr), processedColInd(nullptr), processedRowInd(nullptr),
          processedValues(nullptr) {
    }

    ~SpMSpVHandleSOA() {
        cudaFree(processedColInd);
        delete[] processedRowInd;
        cudaFree(processedValues);
        cudaFree(vectorValuesDevice);
    }

    // 假设为SOA内存布局，并且所有数据存储在主机端
    // 后续可扩展为设备端存储，降低内存操作开销，进一步优化性能
    // [{x_0, x_1, x_2, ..., x_n}, {y_0, y_1, y_2, ..., y_n}, ..., {z_0, z_1, z_2, ..., z_n}]
    // 预处理矩阵，提取其中的有效数据（与输入稀疏向量对应的列）
    // 预处理后的矩阵存储有效数据，以及存储用于在输入向量中定位的相对索引数据
    void setup() {
        cudaMalloc(&vectorValuesDevice, numDims * numVectorRows * sizeof(Real));
        cudaMemcpy(vectorValuesDevice, vectorValues, numDims * numVectorRows * sizeof(Real), cudaMemcpyHostToDevice);
        // Step 1
        // 计算稀疏矩阵中待处理的非零元数量
        for (auto i = 0; i < numVectorRows; ++i) {
            auto col = vectorIndex[i];
            numProcessedNonZero += matrixColPtr[col + 1] - matrixColPtr[col];
        }

        // Step 2
        // 分配存储空间
        // 此处使用统一内存，后续可根据需要调整
        // 行索引使用主机端内存
        cudaMallocManaged(&processedColInd, numProcessedNonZero * sizeof(Integer));
        cudaMallocManaged(&processedValues, numDims * numProcessedNonZero * sizeof(Real));
        processedRowInd = new Integer[numProcessedNonZero];

        // Step 3
        // 提取有效数据，依然按照SOA布局
        // 此处可尝试使用GPU并行处理，但需要辅助数据来存储每列的起始位置
        unsigned int writeIdx = 0;
        for (auto i = 0; i < numVectorRows; ++i) {
            auto col = vectorIndex[i];
            auto colStart = matrixColPtr[col];
            auto colEnd = matrixColPtr[col + 1];

            for (auto j = colStart; j < colEnd; ++j) {
                processedColInd[writeIdx] = i;
                processedRowInd[writeIdx] = matrixRowInd[j];
                // 复制对应维度的值
                for (auto d = 0; d < numDims; ++d) {
                    processedValues[d * numProcessedNonZero + writeIdx] =
                        matrixValues[d * matrixColPtr[numMatrixCols] + j];
                }
                writeIdx++;
            }
        }
    }

    // 计算
    // 假设y也为稀疏向量，并且也为SOA内存布局
    // 假设y尚未分配存储空间
    unsigned int compute(Integer** yIndex, Real** yValue) {
        const auto totalNumNoneZero = numDims * numProcessedNonZero;
        Real* yValueDevice;
        Integer *yIndexDevice;
        cudaMalloc(&yValueDevice, totalNumNoneZero * sizeof(Real));
        cudaMalloc(&yIndexDevice, numProcessedNonZero * sizeof(Integer));
        dim3 block(512);
        dim3 grid((numProcessedNonZero + block.x - 1) / block.x);
        SpMSpVKernelSOA<Integer, Real><<<grid, block>>>(yValueDevice, processedColInd, processedValues, vectorValuesDevice,
                                                        numDims, numProcessedNonZero, numVectorRows);
        cudaDeviceSynchronize();
        cudaMemcpy(*yIndex, processedRowInd, numProcessedNonZero * sizeof(Integer), cudaMemcpyHostToHost);
        yValue = &yValueDevice;
        yIndex = &yIndexDevice;
        return numProcessedNonZero;
    }

private:
    unsigned int numDims;
    unsigned int numMatrixRows;
    unsigned int numMatrixCols;
    unsigned int numVectorRows;
    unsigned int numProcessedNonZero{};
    Integer* matrixColPtr;
    Integer* matrixRowInd;
    Real* matrixValues;
    Integer* vectorIndex;
    Real* vectorValues;
    Real* vectorValuesDevice;
    Integer* processedColInd;
    Integer* processedRowInd;
    Real* processedValues;
};

GridAsSparseMatrix* SpTSpMMultiplication(SparseTensorCscFormat* P, GridAsSparseMatrix* grid);

template <typename Integer, typename Real>
__global__ void SpMSpVKernelAOS_v2(Real* yValue,
                                    const Integer* colInd,           // 输入向量列索引
                                    const Integer* matrixPosInd,     // 原始矩阵位置索引
                                    const Real* matrixData,          // 原始矩阵数据
                                    const Real* xValue,              // 输入向量数据
                                    const unsigned int numDims,
                                    const unsigned int totalNumNoneZero) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = blockDim.x * gridDim.x;

#pragma unroll
    for (auto i = 0; i < UNROLL_FACTOR; ++i) {
        if (const auto index = idx + i * stride; index < totalNumNoneZero) {
            const auto elementIdx = index / numDims;       // 当前元素索引
            const auto dim = index % numDims;              // 当前维度

            const auto col = colInd[elementIdx];           // 输入向量的列
            const auto matPos = matrixPosInd[elementIdx];  // 原始矩阵中的位置

            // 直接从原始矩阵读取数据并计算
            yValue[index] = matrixData[matPos * numDims + dim] *
                           xValue[col * numDims + dim];
        }
    }
}

GridAsSparseMatrix* SpTSpMMultiplication_v2(SparseTensorCscFormat* P, GridAsSparseMatrix* grid);

GridAsSparseMatrix* SpTSpMMultiplication_v3(SparseTensorCscFormat* P, GridAsSparseMatrix* grid, double* d_P_values);

#endif //SPMSPV_SPMSPV_CUH

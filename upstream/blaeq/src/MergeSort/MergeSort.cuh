#ifndef CUDADB_MERGESORT_CUH
#define CUDADB_MERGESORT_CUH

#include <cuda_runtime.h>
#include <limits>
#include <thread>
#include <vector>
#include <cstring>
#include <queue>
#include <mutex>
#include <condition_variable>

enum class MemoryLocation {
    Host,
    Device,
    Unified
};

template <typename T>
__device__ __forceinline__ T cudaMin(T a, T b) {
    return b < a ? b : a;
}

template <typename T>
__device__ __forceinline__ T cudaMax(T a, T b) {
    return a < b ? b : a;
}

/**
 * @brief Generic warp shuffle operation that works with any type by decomposing into 32-bit chunks.
 * 
 * @tparam T The type of the value to shuffle.
 * @param[in] val The value to shuffle.
 * @param[in] srcLane The source lane to read from.
 * @param[in] mask The warp participation mask.
 * @return The shuffled value from the specified source lane.
 */
template <typename T>
__device__ __forceinline__ T shflSync(T val, int srcLane, unsigned mask = 0xffffffff) {
    constexpr int numInts = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    union Converter {
        T value;
        int ints[numInts];
    };

    Converter input;
    Converter output;
    input.value = val;

#pragma unroll
    for (int i = 0; i < numInts; ++i) {
        output.ints[i] = __shfl_sync(mask, input.ints[i], srcLane);
    }

    return output.value;
}

/**
 * @brief Generic warp shuffle XOR operation for any type.
 *
 * @tparam T The type of the value to shuffle.
 * @param[in] val The value to shuffle.
 * @param[in] laneMask The XOR mask to compute target lane.
 * @param[in] mask The warp participation mask.
 * @return The shuffled value from the XOR'd lane.
 */
template <typename T>
__device__ __forceinline__ T shflXorSync(T val, int laneMask, unsigned mask = 0xffffffff) {
    constexpr int numInts = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    union Converter {
        T value;
        int ints[numInts];
    };

    Converter input;
    Converter output;
    input.value = val;

#pragma unroll
    for (int i = 0; i < numInts; ++i) {
        output.ints[i] = __shfl_xor_sync(mask, input.ints[i], laneMask);
    }

    return output.value;
}

/**
 * @brief Performs a compare-and-swap operation between threads in a warp using shuffle intrinsics.
 *
 * @tparam T The type of the value to swap.
 * @tparam Ascending Sort direction: true for ascending, false for descending.
 * @param[in,out] val The value held by the current thread, updated in-place.
 * @param[in] mask The XOR mask to calculate the target lane ID.
 * @param[in] dir Sort direction: true for ascending, false for descending.
 */
template <typename T, bool Ascending>
__device__ __forceinline__ void compareAndSwap(T& val, const int mask, const bool dir) {
    T otherVal = shflXorSync(val, mask);
    bool isLowLane = (threadIdx.x & mask) == 0;
    bool keepVal;

    if (Ascending) {
        if (dir) {
            keepVal = isLowLane ? (val < otherVal) : !(val < otherVal);
        }
        else {
            keepVal = isLowLane ? !(val < otherVal) : (val < otherVal);
        }
    }
    else {
        if (dir) {
            keepVal = isLowLane ? !(val < otherVal) : (val < otherVal);
        }
        else {
            keepVal = isLowLane ? (val < otherVal) : !(val < otherVal);
        }
    }

    if (!keepVal) {
        val = otherVal;
    }
}

/**
 * @brief Sorts a value within a 32-thread warp using a hardcoded bitonic sorting network.
 *
 * @tparam T The type of the value.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param val The value held by the calling thread.
 * @return The value belonging to this thread's rank after sorting the warp.
 */
template <typename T, bool Ascending>
__device__ __forceinline__ T warpSort(T val) {
    int laneId = threadIdx.x & 31;

    compareAndSwap<T, Ascending>(val, 1, (laneId & 2) == 0);
    compareAndSwap<T, Ascending>(val, 2, (laneId & 4) == 0);
    compareAndSwap<T, Ascending>(val, 1, (laneId & 4) == 0);
    compareAndSwap<T, Ascending>(val, 4, (laneId & 8) == 0);
    compareAndSwap<T, Ascending>(val, 2, (laneId & 8) == 0);
    compareAndSwap<T, Ascending>(val, 1, (laneId & 8) == 0);
    compareAndSwap<T, Ascending>(val, 8, (laneId & 16) == 0);
    compareAndSwap<T, Ascending>(val, 4, (laneId & 16) == 0);
    compareAndSwap<T, Ascending>(val, 2, (laneId & 16) == 0);
    compareAndSwap<T, Ascending>(val, 1, (laneId & 16) == 0);
    compareAndSwap<T, Ascending>(val, 16, true);
    compareAndSwap<T, Ascending>(val, 8, true);
    compareAndSwap<T, Ascending>(val, 4, true);
    compareAndSwap<T, Ascending>(val, 2, true);
    compareAndSwap<T, Ascending>(val, 1, true);

    return val;
}

/**
 * @brief Finds the partition point for the merge path algorithm using binary search in shared memory.
 *
 * @tparam T The type of the elements.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param sharedData Pointer to the shared memory buffer containing both arrays.
 * @param startA Starting index of the first sorted sequence (Array A).
 * @param lenA Length of Array A.
 * @param startB Starting index of the second sorted sequence (Array B).
 * @param lenB Length of Array B.
 * @param diag The diagonal index (k-th element) in the merge matrix.
 * @return The index in Array A representing the split point.
 */
template <typename T, bool Ascending>
__device__ __forceinline__ int mergePathIntersectionShared(
    const T* sharedData,
    const int startA, const int lenA,
    const int startB, const int lenB,
    const int diag) {
    int xMin = max(0, diag - lenB);
    int xMax = min(diag, lenA);

    while (xMin < xMax) {
        const int mid = (xMin + xMax) >> 1;
        const int x = mid;
        const int y = diag - mid;

        bool condition;
        if (Ascending) {
            condition = x < lenA && y > 0 && !(sharedData[startB + y - 1] < sharedData[startA + x]);
        }
        else {
            condition = x < lenA && y > 0 && !(sharedData[startA + x] < sharedData[startB + y - 1]);
        }

        if (condition) {
            xMin = mid + 1;
        }
        else {
            xMax = mid;
        }
    }

    return xMin;
}

/**
 * @brief A highly optimized block-level sort kernel using registers and shared memory.
 *
 * @tparam TotalItems Total number of items processed per block (Must be a power of 2 * blockDim).
 * @tparam ItemsPerThread Number of items processed per thread.
 * @tparam T The type of elements to sort.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param[in] inputData Pointer to global input array.
 * @param[out] outputData Pointer to global output array.
 * @param n Total size of the global array.
 * @param paddingVal Value used to pad the array if n is not a multiple of TotalItems.
 */
template <int TotalItems, int ItemsPerThread, typename T, bool Ascending>
__global__ void blockMergeSortKernel(const T* __restrict__ inputData, T* __restrict__ outputData, const int n,
                                     T paddingVal) {
    __shared__ T sharedData[TotalItems];
    T registers[ItemsPerThread];

    const int threadId = threadIdx.x;
    const int blockOffset = blockIdx.x * TotalItems;

#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        int globalIdx = blockOffset + threadId + i * blockDim.x;
        registers[i] = (globalIdx < n) ? inputData[globalIdx] : paddingVal;
        registers[i] = warpSort<T, Ascending>(registers[i]);
    }

#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        sharedData[threadId + i * blockDim.x] = registers[i];
    }
    __syncthreads();

    for (int width = 32; width < TotalItems; width <<= 1) {
        const int mergedSize = width << 1;
        const int maskDiag = mergedSize - 1;

#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            const int destIdx = threadId + i * blockDim.x;
            const int pairIdx = destIdx / mergedSize;
            const int localDiag = destIdx & maskDiag;
            const int startA = pairIdx * mergedSize;
            const int startB = startA + width;

            const int aSplit = mergePathIntersectionShared<T, Ascending>(
                sharedData, startA, width, startB, width, localDiag
            );
            const int bSplit = localDiag - aSplit;

            T valA = (aSplit < width) ? sharedData[startA + aSplit] : paddingVal;
            T valB = (bSplit < width) ? sharedData[startB + bSplit] : paddingVal;

            bool chooseA;
            if (Ascending) {
                chooseA = aSplit < width && (bSplit >= width || !(valB < valA));
            }
            else {
                chooseA = aSplit < width && (bSplit >= width || !(valA < valB));
            }

            if (chooseA) {
                registers[i] = valA;
            }
            else {
                registers[i] = valB;
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            sharedData[threadId + i * blockDim.x] = registers[i];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < ItemsPerThread; ++i) {
        int globalIdx = blockOffset + threadId + i * blockDim.x;
        if (globalIdx < n) {
            outputData[globalIdx] = registers[i];
        }
    }
}

/**
 * @brief Merges two sorted ranges in parallel on the host.
 *
 * @tparam T The type of elements to merge.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param src Source array containing both sorted ranges.
 * @param dst Destination array for merged result.
 * @param start Start index of the first range.
 * @param mid End index of the first range (exclusive), start of the second range.
 * @param end End index of the second range (exclusive).
 */
template <typename T, bool Ascending>
void parallelMerge(const T* src, T* dst, int start, int mid, int end) {
    int i = start;
    int j = mid;
    int k = start;

    while (i < mid && j < end) {
        bool condition = Ascending ? (src[i] <= src[j]) : (src[i] >= src[j]);
        if (condition) {
            dst[k++] = src[i++];
        }
        else {
            dst[k++] = src[j++];
        }
    }

    while (i < mid) {
        dst[k++] = src[i++];
    }
    while (j < end) {
        dst[k++] = src[j++];
    }
}

/**
 * @brief Performs parallel merge sort on the host using multiple threads.
 *
 * @tparam T The type of elements to sort.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param data Array to sort.
 * @param n Size of the array.
 * @param blockSize Size of pre-sorted blocks from GPU.
 */
template <typename T, bool Ascending>
void hostParallelMergeSort(T* data, int n, int blockSize) {
    std::vector<T> temp(n);
    T* src = data;
    T* dst = temp.data();

    for (int width = blockSize; width < n; width *= 2) {
        int numMerges = (n + 2 * width - 1) / (2 * width);
        int numThreads = std::min(numMerges, (int)std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        int mergesPerThread = (numMerges + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; ++t) {
            threads.emplace_back([=, &src, &dst]() {
                int startMerge = t * mergesPerThread;
                int endMerge = std::min(startMerge + mergesPerThread, numMerges);

                for (int m = startMerge; m < endMerge; ++m) {
                    int left = m * 2 * width;
                    int mid = std::min(left + width, n);
                    int right = std::min(left + 2 * width, n);

                    if (mid < right) {
                        parallelMerge<T, Ascending>(src, dst, left, mid, right);
                    }
                    else {
                        for (int i = left; i < mid; ++i) {
                            dst[i] = src[i];
                        }
                    }
                }
            });
        }

        for (auto& th : threads) {
            th.join();
        }

        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, n * sizeof(T));
    }
}

/**
 * @brief Serial merge sort on the host.
 *
 * @tparam T The type of elements to sort.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param data Array to sort.
 * @param n Size of the array.
 * @param blockSize Size of pre-sorted blocks from GPU.
 */
template <typename T, bool Ascending>
void hostSerialMergeSort(T* data, int n, int blockSize) {
    std::vector<T> temp(n);
    T* src = data;
    T* dst = temp.data();

    for (int width = blockSize; width < n; width *= 2) {
        int numMerges = (n + 2 * width - 1) / (2 * width);

        for (int m = 0; m < numMerges; ++m) {
            int left = m * 2 * width;
            int mid = std::min(left + width, n);
            int right = std::min(left + 2 * width, n);

            if (mid < right) {
                parallelMerge<T, Ascending>(src, dst, left, mid, right);
            }
            else {
                for (int i = left; i < mid; ++i) {
                    dst[i] = src[i];
                }
            }
        }

        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, n * sizeof(T));
    }
}

namespace cudaDatabaseMergeSort {
    class GlobalThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queueMutex;
        std::condition_variable condition;
        std::condition_variable completionCondition;
        bool stop;
        int activeTasks;
        int pendingTasks;

        GlobalThreadPool() : stop(false), activeTasks(0), pendingTasks(0) {
            size_t numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4;

            for (size_t i = 0; i < numThreads; ++i) {
                workers.emplace_back([this] {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queueMutex);
                            this->condition.wait(lock, [this] {
                                return this->stop || !this->tasks.empty();
                            });

                            if (this->stop && this->tasks.empty()) {
                                return;
                            }

                            if (!this->tasks.empty()) {
                                task = std::move(this->tasks.front());
                                this->tasks.pop();
                                ++this->activeTasks;
                            }
                        }

                        if (task) {
                            task();
                            {
                                std::lock_guard<std::mutex> lock(this->queueMutex);
                                --this->activeTasks;
                                --this->pendingTasks;
                            }
                            this->completionCondition.notify_all();
                        }
                    }
                });
            }
        }

        ~GlobalThreadPool() {
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                stop = true;
            }
            condition.notify_all();
            for (std::thread& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
        }

    public:
        // 禁止拷贝和赋值
        GlobalThreadPool(const GlobalThreadPool&) = delete;
        GlobalThreadPool& operator=(const GlobalThreadPool&) = delete;

        static GlobalThreadPool& getInstance() {
            static GlobalThreadPool instance;
            return instance;
        }

        template <class F>
        void enqueue(F&& f) {
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                tasks.emplace(std::forward<F>(f));
                ++pendingTasks;
            }
            condition.notify_one();
        }

        void wait() {
            std::unique_lock<std::mutex> lock(queueMutex);
            completionCondition.wait(lock, [this] {
                return this->pendingTasks == 0 && this->activeTasks == 0;
            });
        }
    };
}

/**
 * @brief Thread pool-based merge sort on the host.
 *
 * @tparam T The type of elements to sort.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @param data Array to sort.
 * @param n Size of the array.
 * @param blockSize Size of pre-sorted blocks from GPU.
 */
template <typename T, bool Ascending>
void hostThreadPoolMergeSort(T* data, int n, int blockSize) {
    std::vector<T> temp(n);
    T* src = data;
    T* dst = temp.data();

    cudaDatabaseMergeSort::GlobalThreadPool& pool = cudaDatabaseMergeSort::GlobalThreadPool::getInstance();

    for (int width = blockSize; width < n; width *= 2) {
        int numMerges = (n + 2 * width - 1) / (2 * width);

        for (int m = 0; m < numMerges; ++m) {
            pool.enqueue([=, &src, &dst]() {
                int left = m * 2 * width;
                int mid = std::min(left + width, n);
                int right = std::min(left + 2 * width, n);

                if (mid < right) {
                    parallelMerge<T, Ascending>(src, dst, left, mid, right);
                } else {
                    for (int i = left; i < mid; ++i) {
                        dst[i] = src[i];
                    }
                }
            });
        }

        pool.wait();
        std::swap(src, dst);
    }

    if (src != data) {
        std::memcpy(data, src, n * sizeof(T));
    }
}


/**
 * @brief Detects the memory location of a pointer.
 *
 * @tparam T The type of elements.
 * @param ptr Pointer to check.
 * @return MemoryLocation indicating where the pointer resides.
 */
template <typename T>
MemoryLocation detectMemoryLocation(const T* ptr) {
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);

    if (err != cudaSuccess) {
        cudaGetLastError();
        return MemoryLocation::Host;
    }

    if (attrs.type == cudaMemoryTypeUnregistered) {
        return MemoryLocation::Host;
    }
    else if (attrs.type == cudaMemoryTypeHost) {
        return MemoryLocation::Host;
    }
    else if (attrs.type == cudaMemoryTypeDevice) {
        return MemoryLocation::Device;
    }
    else if (attrs.type == cudaMemoryTypeManaged) {
        return MemoryLocation::Unified;
    }

    return MemoryLocation::Host;
}

/**
 * @brief Host wrapper function for GPU-accelerated sorting with automatic memory management.
 *
 * @tparam TotalItems Total number of items processed per block.
 * @tparam ItemsPerThread Number of items processed per thread.
 * @tparam Ascending Sort order: true for ascending, false for descending.
 * @tparam T The type of elements to sort.
 * @param inputData Pointer to input data (can be host, device, or unified memory).
 * @param outputData Pointer to output data (can be host, device, or unified memory).
 * @param n Number of elements to sort.
 * @param paddingVal Value used for padding if n is not a multiple of TotalItems.
 */
template <int TotalItems, int ItemsPerThread, bool Ascending, typename T>
void gpuSort(const T* inputData, T* outputData, int n, T paddingVal) {
    const int threadsPerBlock = TotalItems / ItemsPerThread;
    const int blocks = (n + TotalItems - 1) / TotalItems;

    MemoryLocation inputLoc = detectMemoryLocation(inputData);
    MemoryLocation outputLoc = detectMemoryLocation(outputData);

    T* d_in = nullptr;
    T* d_out = nullptr;
    T* h_temp = nullptr;

    bool needInputCopy = (inputLoc == MemoryLocation::Host);
    bool needOutputCopy = (outputLoc == MemoryLocation::Host);

    if (needInputCopy) {
        cudaMalloc(&d_in, n * sizeof(T));
        cudaMemcpy(d_in, inputData, n * sizeof(T), cudaMemcpyHostToDevice);
    }
    else {
        d_in = const_cast<T*>(inputData);
    }

    if (needOutputCopy) {
        cudaMalloc(&d_out, n * sizeof(T));
    }
    else {
        d_out = outputData;
    }

    blockMergeSortKernel<TotalItems, ItemsPerThread, T, Ascending>
        <<<blocks, threadsPerBlock>>>(d_in, d_out, n, paddingVal);
    cudaDeviceSynchronize();

    if (needOutputCopy) {
        h_temp = new T[n];
        cudaMemcpy(h_temp, d_out, n * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else {
        h_temp = d_out;
    }

    if (blocks > 1) {
        if (needOutputCopy) {
            hostSerialMergeSort<T, Ascending>(h_temp, n, TotalItems);
            std::memcpy(outputData, h_temp, n * sizeof(T));
        }
        else {
            T* h_final = new T[n];
            cudaMemcpy(h_final, d_out, n * sizeof(T), cudaMemcpyDeviceToHost);
            hostSerialMergeSort<T, Ascending>(h_final, n, TotalItems);
            cudaMemcpy(outputData, h_final, n * sizeof(T), cudaMemcpyHostToDevice);
            delete[] h_final;
        }
    }
    else if (needOutputCopy) {
        std::memcpy(outputData, h_temp, n * sizeof(T));
    }

    if (needInputCopy && d_in) {
        cudaFree(d_in);
    }
    if (needOutputCopy && d_out) {
        cudaFree(d_out);
    }
    if (needOutputCopy && h_temp) {
        delete[] h_temp;
    }
}

#endif

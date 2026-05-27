#include "isin_cuda.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h> // Required for half-precision support

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define THREADS 1024

namespace
{
  // Helper function for binary search (used in sparse mode)
  template <typename scalar_t>
  __device__ bool binary_search(const scalar_t *test_elements, int64_t left, int64_t right, scalar_t target)
  {
    while (left <= right)
    {
      int64_t mid = left + (right - left) / 2;
      if (test_elements[mid] == target)
      {
        return true;
      }
      else if (test_elements[mid] < target)
      {
        left = mid + 1;
      }
      else
      {
        right = mid - 1;
      }
    }
    return false;
  }

  // Sparse mode kernel: Uses binary search for sorted test_elements
  template <typename scalar_t>
  __global__ void isin_cuda_sparse_kernel(
      const scalar_t *__restrict__ elements,
      const scalar_t *__restrict__ test_elements,
      const int64_t batch_size,
      const int64_t elements_size,
      const int64_t test_elements_size,
      const int64_t padding_idx,
      int *output)
  {
    const int batch_index = blockIdx.x;
    const int element_index = threadIdx.x;

    if (element_index >= elements_size) return;

    const scalar_t element = elements[batch_index * elements_size + element_index];
    int *output_ptr = output + batch_index * elements_size + element_index;

    if (element == padding_idx)
      return;

    // Perform binary search on sorted test_elements within each batch
    const scalar_t *test_elements_batch = test_elements + batch_index * test_elements_size;
    bool found = binary_search(test_elements_batch, 0, test_elements_size - 1, element);
    if (found)
    {
      *output_ptr = 1;
    }
  }

  // Dense mode kernel: Directly index into test_elements as a lookup table
  template <typename scalar_t, typename test_scalar_t>
  __global__ void isin_cuda_dense_kernel(
      const scalar_t *__restrict__ elements,
      const test_scalar_t *__restrict__ test_elements,
      const int64_t batch_size,
      const int64_t elements_size,
      const int64_t test_elements_size,
      const int64_t padding_idx,
      int *output)
  {
    const int batch_index = blockIdx.x;
    const int element_index = threadIdx.x;

    if (element_index >= elements_size) return;

    const scalar_t element = elements[batch_index * elements_size + element_index];
    int *output_ptr = output + batch_index * elements_size + element_index;

    if (element == padding_idx)
        return;

    // Cast element to int32_t for safe indexing
    int32_t element_int = static_cast<int32_t>(element);

    // Dense lookup: test_elements is treated as a [B, test_elements_size] lookup table
    if (element_int < test_elements_size && static_cast<int>(test_elements[batch_index * test_elements_size + element_int]) != 0)
    {
        *output_ptr = 1;
    }
  }
} // namespace

at::Tensor isin_cuda(
    at::Tensor elements,
    at::Tensor test_elements,
    int64_t padding_idx,
    bool dense_lookup)
{
  CHECK_CUDA(elements);
  CHECK_CUDA(test_elements);
  cudaSetDevice(elements.get_device());

  const int64_t batch_size = elements.size(0);
  const int64_t elements_size = elements.size(1);
  const int64_t test_elements_size = test_elements.size(1);

  auto output = at::zeros({batch_size, elements_size}, elements.options().dtype(at::kInt));

  // Configure grid and block dimensions
  dim3 block(THREADS);
  dim3 grid(batch_size);

  AT_DISPATCH_ALL_TYPES(elements.scalar_type(), "isin_cuda", ([&] {
    if (dense_lookup) {
      // Manually check test_elements type and dispatch the dense kernel accordingly
      if (test_elements.scalar_type() == at::kFloat) {
        isin_cuda_dense_kernel<scalar_t, float><<<grid, block>>>(
            elements.data_ptr<scalar_t>(),
            test_elements.data_ptr<float>(),
            batch_size,
            elements_size,
            test_elements_size,
            padding_idx,
            output.data_ptr<int>());
      } else if (test_elements.scalar_type() == at::kHalf) {
        isin_cuda_dense_kernel<scalar_t, at::Half><<<grid, block>>>(
            elements.data_ptr<scalar_t>(),
            test_elements.data_ptr<at::Half>(),
            batch_size,
            elements_size,
            test_elements_size,
            padding_idx,
            output.data_ptr<int>());
      } else {
        TORCH_CHECK(false, "Unsupported test_elements type for dense lookup");
      }
    } else {
      // Call sparse kernel
      isin_cuda_sparse_kernel<scalar_t><<<grid, block>>>(
          elements.data_ptr<scalar_t>(),
          test_elements.data_ptr<scalar_t>(),
          batch_size,
          elements_size,
          test_elements_size,
          padding_idx,
          output.data_ptr<int>());
    }
  }));

  return output.toType(at::kBool);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("isin_cuda", &isin_cuda, "IsIn CUDA kernel with padding and dense/sparse lookup selection");
}

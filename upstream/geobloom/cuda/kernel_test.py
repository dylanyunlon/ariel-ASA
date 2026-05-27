import torch
from torch.utils.cpp_extension import load

# Compile and load the CUDA extension with Ninja
isin_cuda = load(
    name="isin_cuda",
    sources=["cuda/isin_cuda.cu"],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-lineinfo'],
    verbose=True
)

# Define test parameters
elements = torch.tensor([1, 2, 0, 0, 5, 7], device="cuda", dtype=torch.int64).view(2, 3)
test_elements = torch.tensor([3, 0, 1, 10, 10, 7, 2, 4], device="cuda", dtype=torch.int64).view(2, 4)
padding_idx = 0

# Run the kernel
output = isin_cuda.isin_cuda(elements, test_elements, padding_idx, False)

test_elements_dense = torch.zeros((2, 8), device="cuda", dtype=torch.float32)
test_elements_dense[0, 1] = 1
test_elements_dense[1, 5] = 1
output_dense = isin_cuda.isin_cuda(elements, test_elements_dense, padding_idx, True)

# Print output
print("Elements:", elements)
print("Test Elements:", test_elements)
print("Test Elements Dense:", test_elements_dense)
print("Output:", output)
print("Output Dense:", output_dense)
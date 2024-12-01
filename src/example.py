import torch

import sys

sys.path.append("cuda_extension.so")
import cuda_extension

# Example usage.
input_tensor = torch.randn(10, device="cuda").bfloat16()
print(f"before: {input_tensor}")
cuda_extension.my_cuda_function(input_tensor)
print(f"after: {input_tensor}")

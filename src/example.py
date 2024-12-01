import torch

import sys

sys.path.append("cuda_extension.so")
import cuda_extension

# Example usage.
a = torch.randn(10, device="cuda").bfloat16()
b = torch.randn(10, device="cuda").bfloat16()
print(f"a: {a}")
print(f"b: {b}")
cuda_extension.add(a, b)
print(f"after: {a}")

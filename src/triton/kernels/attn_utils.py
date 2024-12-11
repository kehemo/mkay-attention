import torch 
from typing import Optional, Sequence, Tuple, Union
import math

def force_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def compute_relative_rmse(t1: torch.Tensor, t2: torch.Tensor, drop_dims = []) -> torch.Tensor:
    """
    t1 is ground truth
    t2 is computed version
    """
    assert t1.shape == t2.shape, f"Tensor shapes must match: {t1.shape} != {t2.shape}"
    for dim in drop_dims:
        t1 = t1.index_select(dim, torch.tensor([0], device=t1.device) )
        t2 = t2.index_select(dim, torch.tensor([0], device=t2.device) )

    t1_flat = t1.reshape(-1)
    t2_flat = t2.reshape(-1)

    diff = t1_flat - t2_flat
    mse = torch.mean(diff * diff)
    ref_mse = torch.mean(t1_flat * t1_flat)

    rmse = torch.sqrt(mse)
    ref_rmse = torch.sqrt(ref_mse)
    rel_rmse = rmse / ref_rmse

    return rel_rmse

def tensor_to_txt(result, output_name):
    with open(f"{output_name}.txt", "w") as file:
        # Iterate over rows of the tensor
        for row in result.cpu().numpy():  # Move to CPU and convert to NumPy
            file.write(" ".join(map(str, row)) + "\n")  # Write each row as a space-separated string

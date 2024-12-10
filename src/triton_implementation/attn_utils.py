import torch 
from typing import Optional, Sequence, Tuple, Union
import math

def force_contiguous(x):
    print("BE VERY CAREFUL - THE FORCE CONTIGUOUS FUNCTION (PROBABLY) DOES NOT DO WHAT YOU THINK IT DOES")
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def compute_relative_rmse(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    t1 is ground truth
    t2 is computed version
    """
    assert t1.shape == t2.shape, f"Tensor shapes must match: {t1.shape} != {t2.shape}"

    t1_flat = t1.reshape(-1)
    t2_flat = t2.reshape(-1)

    diff = t1_flat - t2_flat
    mse = torch.mean(diff * diff)
    ref_mse = torch.mean(t1_flat * t1_flat)

    rmse = torch.sqrt(mse)
    ref_rmse = torch.sqrt(ref_mse)
    rel_rmse = rmse / ref_rmse

    return rel_rmse


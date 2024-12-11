import torch
import triton
import triton.language as tl
from triton.runtime import driver

import math
from naive_reference import naive_forward_batched_supports
from attn_utils import compute_relative_rmse, tensor_to_txt

from fwd_FA2_bf16 import flash2_fwd_wrapper as fwd_FA2_bf16_wrapper
from fwd_FA1_fp32 import flash1_fwd_wrapper as fwd_FA1_fp32_wrapper

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""

def attention_fwd_eval(qkv, target_wrapper, target_dtype = torch.float32, name = "FA2"):
    q, k, v = qkv.unbind(dim=2)
    B, N, H, D = q.shape

    sm_scale = 1 / math.sqrt(D)
    S_naive_fwd, P_naive_fwd, O_naive_fwd, L_naive_fwd, l_naive_fwd, m_naive_fwd = naive_forward_batched_supports(q, k, v, sm_scale)

    """TRITON"""
    q = q.to(target_dtype)
    k = k.to(target_dtype)
    v = v.to(target_dtype)

    O_flash = target_wrapper(q, k, v, sm_scale)
    O_flash = O_flash.to(torch.float32)

    """RESULTS"""

    print(f"\nRESULTS FOR {name}")
    print("=====================================")
    print("triton / naive O RMSE: ", compute_relative_rmse(O_flash, O_naive_fwd))
    O_correct = torch.allclose(O_flash, O_naive_fwd, rtol=5e-1, atol=1e-2)
    print(f"Correctness for {name}:")
    print(f"O_correct: {O_correct}")



if __name__ == "__main__":
    torch.manual_seed(0)


    batch_size = 1
    seq_len = 256
    n_heads = 32
    head_dim = 64
    qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device = 'cuda', dtype=torch.float32)
    
    
    # attention_fwd_eval(qkv, attention_triton_launch, name = "Flash1 for FP32")
    attention_fwd_eval(qkv, fwd_FA1_fp32_wrapper, name = "Flash1 for FP32")
    attention_fwd_eval(qkv, fwd_FA2_bf16_wrapper, target_dtype = torch.bfloat16, name = "Flash1 for BF16")

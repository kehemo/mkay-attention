import torch
import os
import math
from einops import rearrange
from ctypes import *
import sys
from triton_implementation import *


def attention_torch_checkpoint(qkv):
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(d)

    attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale
    attention = torch.softmax(attn_scores, dim=-1)  # softmax along key dimension
    output = torch.einsum("b h t s, b s h d -> b t h d", attention, v)
    # output = attention

    return output.to(dtype=qkv.dtype)


def get_tensor_difference(t1, t2):
    diff = (t1 - t2).to(float)  # torch.quantile() doesn't work in bf16 :(
    rtol = diff / t1
    rtol = rtol.abs().flatten()

    print("========================")
    print(">>>    rtol stats       ")
    print(f"rtol(50%) = {torch.quantile(rtol, 0.5)}")
    print(f"rtol(75%) = {torch.quantile(rtol, 0.75)}")
    print(f"rtol(90%) = {torch.quantile(rtol, 0.90)}")
    print(f"rtol(95%) = {torch.quantile(rtol, 0.95)}")
    print(f"rtol(99%) = {torch.quantile(rtol, 0.99)}")


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


file_path = os.path.realpath(__file__)
script_dir = os.path.dirname(file_path)
telerun_dir = os.path.join(script_dir, "..", "telerun-out")

def from_file(fname, dtype, dims):
        with open(fname, "rb") as f:
            data = f.read()
            buf = (c_char * len(data)).from_buffer_copy(data)
            return torch.frombuffer(buf, dtype=dtype).reshape(dims)

def check(batch_size, seqlen, nheads, headdim, run_id):

    dtype = torch.bfloat16

    test_name = f"test_{batch_size}x{seqlen}x{nheads}x{headdim}"
    prefix = os.path.join(script_dir, test_name)

    o_fname = os.path.join(telerun_dir, run_id, f"{test_name}_o.bin")
    triton_o_fname = os.path.join(telerun_dir, run_id, f"triton_{test_name}_o.bin")
    q_fname = f"{prefix}_q.bin"
    k_fname = f"{prefix}_k.bin"
    v_fname = f"{prefix}_v.bin"
    q, k, v = map(lambda filename: from_file(filename, dtype, dims = (batch_size, seqlen, nheads, headdim)), [q_fname, k_fname, v_fname])
    qkv = torch.stack((q, k, v), dim=2)
    print("\n\n")
    print("=================================================")
    print("Computing batched Q @ K.T")
    print("=================================================")
    print(f"problem size:")
    print(f"q/k/v shape = {(batch_size, seqlen, nheads, headdim)}")
    torch_output = attention_torch_checkpoint(qkv)
    cuda_output = from_file(o_fname, dtype, dims=(batch_size, seqlen, nheads, headdim))
    triton_output = from_file(triton_o_fname, dtype = torch.float32, dims=(batch_size, seqlen, nheads, headdim))
    triton_output = triton_output.to(dtype=torch.bfloat16)

    print("CUDA output compared to Torch output:")
    get_tensor_difference(torch_output, cuda_output)
    rel_rmse = compute_relative_rmse(torch_output, cuda_output)
    print(f"\n\n>>> Relative RMSE: {rel_rmse}")

    print("Triton output compared to Torch output:")
    get_tensor_difference(torch_output, triton_output)
    rel_rmse = compute_relative_rmse(torch_output, triton_output)
    print(f"\n\n>>> Relative RMSE: {rel_rmse}")

run_id = sys.argv[1]
check(2, 32, 64, 32, run_id)

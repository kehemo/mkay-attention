from numpy import ceil
import torch
import os
import math
from einops import rearrange
from ctypes import *
import sys

"""
Backward Pass Implementation Strategy:
- Very liable to messing up a first pass correct implementation of the backward pass.
    - Also kind of tricky dealing with the batching and the multiple heads.
- To help debug, we can use a hierarchy of slightly simpler implementations that can 
  be tested against simpler ones (e.g., by comparing intermediate values), until we
  have one that can be tested directly against the `.backward(g)` Pytorch builtin.

Hierarchy of implementations
- 

"""
sram_size_bytes = 100_000

def fwd_reference(qkv):
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(d)

    attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale
    attention = torch.softmax(attn_scores, dim=-1)  # softmax along key dimension
    output = torch.einsum("b h t s, b s h d -> b t h d", attention, v)
    # output = attention

    # O, S, P
    # O: b t h d
    # S: b h t s (but s is pre-softmax)
    # P: b h t s
    return output.to(dtype=qkv.dtype), attn_scores, attention

def attention_torch_checkpoint(qkv):
    output, *_ = fwd_reference(qkv)
    return output

# Pytorch implementation of the naive backward pass.
# - can *test against* the .backward() of the fwd reference.
# - can *be tested against* for the Pytorch implementation of the backward pass 
def bwd_expanded(
        qkv: torch.Tensor,
        o: torch.Tensor,
        s: torch.Tensor,
        p: torch.Tensor,
        g: torch.Tensor):
    qkv = qkv.detach().clone().requires_grad_(True)
    o, s, p = fwd_reference(qkv)
    q, k, v = qkv.unbind(dim=2)
    
    dV = p.T @ g  # g is dO
    dP = g @ v.T

    # Calculate ds
    # Subtraction factor

    # Einsums my arch nemesis ✊✊✊✊
    dSf = torch.einsum("-> ", p, dP)
    dS = torch.einsum("", )
    raise NotImplementedError

def test_bwd_expanded(qkv: torch.Tensor):
    qkv = qkv.detach().clone().requires_grad_(True)

    o, s, p = fwd_reference(qkv)  # Technically a waste of computation.
    g = torch.randn_like(o)  # Fake loss function
    qkv.backward(g)

    ref = qkv.grad

    # Note for later versions we can also start extracting intermediate results.
    expanded = bwd_expanded(qkv, o, s, p, g)

    get_tensor_difference(ref, expanded)
    rel_rmse = compute_relative_rmse(ref, expanded)
    print(f"\n\n>>> Relative RMSE: {rel_rmse}")

# Pytorch implementation of the FlashAttention backward pass.
# - can *test against* the `bwd_expanded` with intermediate values.
# - can *be tested against* for the CUDA implementation of the backward pass
def bwd_flash(qkv: torch.Tensor):
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    l, m = 0, 0
    B_c = ceil(sram_size_bytes / (4 * d))
    B_r = min(B_c, d)

    T_r = ceil(seqlen / B_r)  # Not affected by num heads or batch_size
    T_c = ceil(seqlen / B_c)

    assert seqlen == T_r * B_r, f"N ({seqlen}) must equal T_r * B_r ({T_r * B_r})"

    Qs = rearrange(q, 'b (t r) h d -> t b r h d', t=T_r, r=B_r)
    Ks = rearrange(k, 'b (t c) h d -> t b c h d', t=T_c, r=B_c)
    Vs = rearrange(v, 'b (t c) h d -> t b c h d', t=T_c, r=B_c)

    # Step 5
    dqs = torch.zeros_like(Qs)
    dks = torch.zeros_like(Ks)
    dvs = torch.zeros_like(Vs)

    for j in range(T_c):
        dKw = torch.zeros_like(Ks) # I don't know convention for the swiggle on top
        dVw = torch.zeros_like(Vs)
        for j in range(T_r):
            s = torch.einsum("")

    raise NotImplementedError

# First pass at the backward pass, in Pytorch first.
# This is deterministic. If we were to do dropout and stuff, we'd the p_drop and rand_state too.
def attention_flash_attn_bwd(qkv, o, do, l, m):
    """
    Inputs:
        QKV in their usual shape, N x d
        O, our ouput              N x d
        dO, our "output gradient" N x d (But this is actually simulated)
        l, statistics             N
        m, statistics             N
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(d)
    raise NotImplementedError


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


def check(batch_size, seqlen, nheads, headdim, run_id):

    dtype = torch.bfloat16

    def from_file(fname, dims=(batch_size, seqlen, nheads, headdim)):
        with open(fname, "rb") as f:
            data = f.read()
            buf = (c_char * len(data)).from_buffer_copy(data)
            return torch.frombuffer(buf, dtype=dtype).reshape(dims)

    test_name = f"test_{batch_size}x{seqlen}x{nheads}x{headdim}"
    prefix = os.path.join(script_dir, test_name)

    o_fname = os.path.join(telerun_dir, run_id, f"{test_name}_o.bin")
    q_fname = f"{prefix}_q.bin"
    k_fname = f"{prefix}_k.bin"
    v_fname = f"{prefix}_v.bin"
    q, k, v = map(from_file, [q_fname, k_fname, v_fname])
    qkv = torch.stack((q, k, v), dim=2)

    print("\n\n")
    print("=================================================")
    print("Computing batched Q @ K.T")
    print("=================================================")
    print(f"problem size:")
    print(f"q/k/v shape = {(batch_size, seqlen, nheads, headdim)}")
    torch_output = attention_torch_checkpoint(qkv)
    cuda_output = from_file(o_fname, dims=(batch_size, seqlen, nheads, headdim))
    get_tensor_difference(torch_output, cuda_output)

    rel_rmse = compute_relative_rmse(torch_output, cuda_output)
    print(f"\n\n>>> Relative RMSE: {rel_rmse}")


run_id = sys.argv[1]
check(2, 32, 64, 32, run_id)

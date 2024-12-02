import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import cuda_extension
import time


def attention_pytorch(qkv, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    softmax_scale = 1.0 / math.sqrt(d)


    attn_score_shape = (batch_size * nheads, seqlen, seqlen)
    attn_scores = torch.empty(attn_score_shape, dtype = attn_dtype, device = attn_device)


    """
    ========================================
    Implementation fork
    - they have a long messy way of computing things
    - i think einsum is a little easier to read here, but its up to whatever you like

    NOTE: softmax_scale =/= 1.0 introduces small tolerance issues.
    If you set it to 1.0, the results will be exactly the same... weird.
    ========================================
    """
    
    # technically they are the ground truth >:(
    use_approach_1 = True 
    if use_approach_1:
        # approach 1
        attn_scores_re = attn_scores.clone().detach()
        
        q_re = rearrange(q, "b t h d -> (b h) t d")
        k_re = rearrange(k, "b s h d -> (b h) d s")

        attn_scores_re = torch.baddbmm(attn_scores_re, q_re, k_re, beta = 0, alpha = softmax_scale)
        attn_scores_re = rearrange(attn_scores_re, "(b h) t s -> b h t s", h = nheads)

        attn_scores = attn_scores_re
    else: 
        # approach 2
        attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale

    attention = torch.softmax(attn_scores, dim = -1) # softmax along key dimension
    output = torch.einsum("b h t s, b s h d -> b t h d", attention, v)

    return output.to(dtype=qkv.dtype)

def attention_torch_checkpoint(qkv):
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(d)

    attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) # * softmax_scale
    output = attn_scores # for now

    return output.to(dtype=qkv.dtype)

def attention_cuda(qkv):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    # q, k, v are of shape (batch_size, seqlen, nheads, head_dim)
    output = cuda_extension.attention_forward(q, k, v)[0]

    print(f"q.shape: {q.shape}")
    print(f"q.stride(): {q.stride()}")
    print(f"qkv.stride(): {qkv.stride()}")
    print(f"cuda output.stride(): {output.stride()}")

    return output

def get_tensor_difference(t1, t2):
    diff = (t1 - t2).to(float) # torch.quantile() doesn't work in bf16 :(
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

def time_fwd(attn_func, qkv):
    time_start = time.time()
    attn_output = attn_func(qkv)
    time_end = time.time()
    time_elapsed = time_end - time_start
    return time_elapsed, attn_output

if __name__ == "__main__":

    batch_size = 8192
    seqlen = 64

    headdim = 64 # 64
    nheads = 32 # 32

    dtype = torch.bfloat16
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device="cuda")


    print("\n\n")
    print("=================================================")
    print("Computing batched Q @ K.T")
    print("=================================================")


    torch_time, torch_output = time_fwd(attention_torch_checkpoint, qkv)
    cuda_time, cuda_output = time_fwd(attention_cuda, qkv)


    print(f"problem size:")
    print(f"q/k/v shape = {(batch_size, seqlen, nheads, headdim)}")
    print(f"torch_time: {torch_time}")
    print(f"cuda_time: {cuda_time}")

    # print("torch_output (ground truth):")
    # print(torch_output)
    # print("cuda_output:")
    # print(cuda_output)

    # get_tensor_difference(torch_output, cuda_output)
    
    rel_rmse = compute_relative_rmse(torch_output, cuda_output)
    print(f"\n\n>>> Relative RMSE: {rel_rmse}")


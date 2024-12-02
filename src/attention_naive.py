import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
# import cuda_extension
import time
import triton
import triton.language as tl


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

    attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale
    attention = torch.softmax(attn_scores, dim = -1) # softmax along key dimension
    output = torch.einsum("b h t s, b s h d -> b t h d", attention, v)
    # output = attention

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

import torch
import triton
import triton.language as tl

@triton.jit
def attention_triton(
    q, k, v, output, seqlen, d,
    batch_stride, seqlen_stride, nheads_stride, nheads, BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # pointers to the appropriate batch and head
    q_ptr = q + batch_idx * batch_stride + head_idx * nheads_stride
    k_ptr = k + batch_idx * batch_stride + head_idx * nheads_stride
    v_ptr = v + batch_idx * batch_stride + head_idx * nheads_stride
    out_ptr = output + batch_idx * batch_stride + head_idx * nheads_stride

    # need to grab the sequences for this particular head
    offs_n = tl.arange(0, BLOCK_SIZE) * seqlen_stride
    offs_d = tl.arange(0, BLOCK_SIZE)

    # shape of q, k, v is (batch_size, seqlen, nheads, d)
    # the block needs to be of shape (seqlen, d) for a single head as described by head_idx
    q_ptrs = q_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]
    k_ptrs = k_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]
    v_ptrs = v_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]

    q_block = tl.load(
        q_ptrs, mask = offs_n[:, None] < seqlen and offs_d[None, :] < d, other = 0
    )
    k_block = tl.load(
        k_ptrs, mask = offs_n[:, None] < seqlen and offs_d[None, :] < d, other = 0
    )
    v_block = tl.load(
        v_ptrs, mask = offs_n[:, None] < seqlen and offs_d[None, :] < d, other = 0
    )

    attn_scores = tl.dot(q_block, tl.trans(k_block)) / tl.sqrt(d.to(tl.float32))

    attn_weights = tl.softmax(attn_scores)

    output_block = tl.dot(attn_weights.to(tl.bfloat16), v_block)

    # Store result
    tl.store(
        out_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :],
        output_block,
        mask=(offs_n[:, None] < seqlen) and (offs_d[None, :] < d)
    )

def attention_triton_launch(qkv):
    batch_size, seqlen, _, nheads, d = qkv.shape

    q, k, v = qkv.unbind(dim=2)

    # Define grid dimensions based on batch size and number of heads
    grid = (batch_size, nheads)


    q_cont = q.contiguous()
    k_cont = k.contiguous()
    v_cont = v.contiguous()

    output = torch.empty_like(q_cont, dtype=torch.bfloat16)
    batch_stride = seqlen * nheads * d
    seqlen_stride = nheads * d
    nheads_stride = d

    attention_triton[grid](
        q_cont, k_cont, v_cont, output,
        seqlen, d, batch_stride, seqlen_stride, nheads_stride, nheads, BLOCK_SIZE=32
    )
    
    output = output.view(batch_size, seqlen, nheads, d)
    return output



def get_tensor_difference(t1, t2):
    diff = (t1 - t2).to(float) # torch.quantile() doesn't work in bf16 :(
    rtol = diff / t1
    rtol = rtol.abs().flatten()
    
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


def run_tester(batch_size, seqlen, headdim, nheads):
    dtype = torch.bfloat16
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device="cuda")

    print("\n======================================")
    print(f"problem size:")
    print(f"q/k/v shape = {(batch_size, seqlen, nheads, headdim)}")
    assert seqlen % 32 == 0, "seqlen must be a multiple of 32 for tiling reasons"


    torch_time, torch_output = time_fwd(attention_torch_checkpoint, qkv)
    # cuda_time, cuda_output = time_fwd(attention_cuda, qkv)
    triton_time, triton_output = time_fwd(attention_triton_launch, qkv)

    # RUN TESTS
    print("-------------------")
    print("runtime checks:")
    print(f"torch_time: {torch_time}")
    # print(f"cuda_time: {cuda_time}")
    print(f"triton_time: {triton_time}")

    # print("torch_output (ground truth):")
    # print(torch_output)
    # print("cuda_output:")
    # print(cuda_output)

    print("-------------------")
    print("correctness checks:")
    use_rtol = True
    if (batch_size < 256 and use_rtol):
        get_tensor_difference(torch_output, triton_output)
    
    rel_rmse = compute_relative_rmse(torch_output, triton_output)
    print(f"\n>>> Relative RMSE: {rel_rmse}")

    del torch_output
    del triton_output
    torch.cuda.empty_cache()

if __name__ == "__main__":

    print("\n\n")
    print("================================================================")
    print("================================================================")
    print("Computing batched Q @ K.T")
    print("================================================================")
    


    # run_tester(batch_size=1, seqlen=32, headdim=4, nheads=1)
    run_tester(batch_size=1, seqlen=32, headdim=32, nheads=1)
    run_tester(batch_size=128, seqlen=64, headdim=64, nheads=32)
    # run_tester(batch_size=1, seqlen=4096, headdim=64, nheads=32)
    # can't currently exceed 4096 due to memory constraints and the inefficiency of the current implementation
    

import torch
import triton
import triton.language as tl
from triton.runtime import driver
import time
import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""


def get_cuda_autotune_config():
    # num_stages_list = [4, 8]
    Br_list = [32]
    Bc_list = [32]
    configs = []
    for Br in Br_list:
        for Bc in Bc_list:
            configs.append(triton.Config({'Br': Br, 'Bc': Bc}))
    return configs
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['seqlen', 'hdim'],
)
@triton.jit
def flash2_fwd(
    q, k, v, 
    output, softmax_score,
    qkv_size, seqlen, nheads, hdim: tl.constexpr,
    L, 
    L_batch_stride, L_heads_stride,
    batch_stride, seqlen_stride, nheads_stride, 
    Br : tl.constexpr, Bc : tl.constexpr):
    # compiler hints, i am unsure how to use these
    # tl.max_constancy(q, qkv_size)
    # tl.max_constancy(k, qkv_size)
    # tl.max_constancy(v, qkv_size)
    # tl.max_contiguous(q, qkv_size)
    # tl.max_contiguous(k, qkv_size)
    # tl.max_contiguous(v, qkv_size)

    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    Tr_i = tl.program_id(2)

    # point to (batch_id, 0, head_id, 0)
    q_ptr = q + batch_id * batch_stride + head_id * nheads_stride
    k_ptr = k + batch_id * batch_stride + head_id * nheads_stride
    v_ptr = v + batch_id * batch_stride + head_id * nheads_stride
    out_ptr = output + batch_id * batch_stride + head_id * nheads_stride
    
    # using make_block_ptr as opposed to just constructing the n-dimensional tensor of pointers because it's easier
    # and also idk how I would mask it out otherwise. i think block ptrs take care of that for you
    k_block_ptr = tl.make_block_ptr(
        base = k_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1), # we can assume stride for d is 1 because we called contiguous() before
        offsets = (0, 0), 
        block_shape = (Bc, hdim),
        order = (1, 0) # this doesn't really matter for correctness unless we're on hopper (jesus christ this is not documented)
        # https://www.mengyibai.com/p/order-in-triton-make-block-ptr/
    )

    v_block_ptr = tl.make_block_ptr(
        base = v_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Bc, hdim),
        order = (1, 0)
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (Tr_i * Br, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    out_block_ptr = tl.make_block_ptr(
        base = out_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (Tr_i * Br, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    # set up L
    L_ptr = L + batch_id * L_batch_stride + head_id * L_heads_stride
    L_offsets = Tr_i * Br + tl.arange(0, Br)
    L_mask = L_offsets < seqlen
    L_ptrs = L_ptr + L_offsets
    
    # set up li, mi
    li = tl.zeros((Br,), dtype=tl.float32)
    mi = tl.full((Br,), -1e9, dtype=tl.float32)

    Qi = tl.load(q_block_ptr)
    Oi = tl.zeros((Br, hdim), dtype=tl.float32)
    Tc = tl.cdiv(seqlen, Bc)

    for j in range(Tc):
        Kj = tl.load(k_block_ptr)
        Vj = tl.load(v_block_ptr)
        
        # Sij: (Br, Bc)
        Sij = (tl.dot(Qi, tl.trans(Kj)) * softmax_score)
        mi_new = tl.maximum(mi, tl.max(Sij, axis = 1))
        P_t_ij = tl.exp(Sij - mi_new[:, None])
        P_t_ij = P_t_ij.to(tl.bfloat16)
        li_new = tl.exp(mi - mi_new) * li + tl.sum(P_t_ij, axis = 1)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi
        Oi = tl.dot(P_t_ij, Vj, Oi) 

        # advance the block pointers
        k_block_ptr = tl.advance(k_block_ptr, (Bc, 0))
        v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))
        mi = mi_new
        li = li_new
        
    Oi = (1 / li[:, None]) * Oi
    Oi = Oi.to(tl.bfloat16)
    Li = mi + tl.log(li)
    Li = Li.to(tl.bfloat16)
    tl.store(out_block_ptr, Oi)
    tl.store(L_ptrs, Li, mask = L_mask)

def flash2_fwd_wrapper(q, k, v, sm_scale):
    q_cont, k_cont, v_cont = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.zeros_like(q_cont, dtype=q.dtype)

    B, N, H, D = q_cont.shape
    qkv_size = q.numel()
    L = torch.zeros((B, H, N), dtype=q.dtype, device=q.device)

    L_batch_stride, L_heads_stride, _ = L.stride()
    batch_stride, seqlen_stride, nheads_stride, _ = q_cont.stride()


    def grid(META):
        Tr = triton.cdiv(N, META['Br'])
        return (B, H, Tr)

    # compile the kernel
    flash2_fwd[grid](
        q_cont, k_cont, v_cont, 
        output, sm_scale,
        qkv_size, N, H, D,
        L,
        L_batch_stride, L_heads_stride,
        batch_stride, seqlen_stride, nheads_stride,
    )

    return output, L
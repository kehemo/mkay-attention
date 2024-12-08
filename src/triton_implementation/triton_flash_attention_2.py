import torch
import triton
import triton.language as tl
import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""


def get_cuda_autotune_config():
    return [
        triton.Config({'Br': 32, 'Bc': 32}, num_stages=4, num_warps=8),
    ]
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['seqlen', 'hdim'],
)
@triton.jit
def attention_triton(
    q, k, v, 
    output, softmax_score,
    seqlen, nheads, hdim: tl.constexpr,
    L, 
    L_batch_stride, L_heads_stride,
    batch_stride, seqlen_stride, nheads_stride, 
    Br : tl.constexpr, Bc : tl.constexpr):

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
    mi = tl.zeros((Br,), dtype=tl.float32) - float("inf")

    Qi = tl.load(q_block_ptr)
    Oi = tl.zeros((Br, hdim), dtype=tl.float32)
    Tc = tl.cdiv(seqlen, Bc)

    # handle j = 0 separately
    Kj = tl.load(k_block_ptr)
    Vj = tl.load(v_block_ptr)

    Sij = tl.dot(Qi, tl.trans(Kj)) * softmax_score
    mi = tl.max(Sij, axis = 1)
    P_t_ij = tl.exp(Sij - mi[:, None])
    li = tl.sum(P_t_ij, axis = 1)
    Oi = tl.dot(P_t_ij, Vj)
    # advance the block pointers
    k_block_ptr = tl.advance(k_block_ptr, (Bc, 0))
    v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))


    for j in range(1, Tc):
        Kj = tl.load(k_block_ptr)
        Vj = tl.load(v_block_ptr)
        
        # Sij: (Br, Bc)
        Sij = tl.dot(Qi, tl.trans(Kj)) * softmax_score
        mi_new = tl.maximum(mi, tl.max(Sij, axis = 1))
        P_t_ij = tl.exp(Sij - mi_new[:, None])
        li_new = tl.exp(mi - mi_new) * li + tl.sum(P_t_ij, axis = 1)
        Oi = (1 / tl.exp(mi - mi_new))[:, None] * Oi + tl.dot(P_t_ij, Vj) 



        # advance the block pointers
        k_block_ptr = tl.advance(k_block_ptr, (Bc, 0))
        v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))
        mi = mi_new
        li = li_new
        
    Oi = (1 / li[:, None]) * Oi
    Li = mi + tl.log(li)
    tl.store(out_block_ptr, Oi)
    tl.store(L_ptrs, Li, mask = L_mask)

def attention_triton_launch(qkv):
    qkv = qkv.to(torch.float32)
    q, k, v = qkv.unbind(dim=2)
    q_cont, k_cont, v_cont = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.zeros_like(q_cont, dtype=qkv.dtype)

    B, N, H, D = q_cont.shape
    L = torch.zeros((B, H, N), dtype=qkv.dtype, device=qkv.device)

    L_batch_stride, L_heads_stride, _ = L.stride()
    batch_stride, seqlen_stride, nheads_stride, _ = q_cont.stride()


    def grid(META):
        Tr = triton.cdiv(N, META['Br'])
        return (B, H, Tr)


    attention_triton[grid](
        q_cont, k_cont, v_cont, 
        output, 1 / math.sqrt(D),
        N, H, D,
        L,
        L_batch_stride, L_heads_stride,
        batch_stride, seqlen_stride, nheads_stride,
    )

    return output.to(torch.bfloat16)
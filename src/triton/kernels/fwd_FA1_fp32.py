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
    l, m, 
    lm_batch_stride, lm_heads_stride,
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

    # set up l_i, m_i
    l_ptr = l + batch_id * lm_batch_stride + head_id * lm_heads_stride
    m_ptr = m + batch_id * lm_batch_stride + head_id * lm_heads_stride
    lm_offsets = Tr_i * Br + tl.arange(0, Br)
    lm_mask = lm_offsets < seqlen
    l_ptrs = l_ptr + lm_offsets
    m_ptrs = m_ptr + lm_offsets



    """
    Stage 2: compute O_i
    """
    Qi = tl.load(q_block_ptr)
    Oi = tl.zeros((Br, hdim), dtype=tl.float32) # TODO: fix this later
    li = tl.load(l_ptrs, mask = lm_mask, other = 0.0)
    mi = tl.load(m_ptrs, mask = lm_mask, other = float("-inf"))


    Tc = tl.cdiv(seqlen, Bc)
    for j in range(0, Tc):
        Kj = tl.load(k_block_ptr)
        Vj = tl.load(v_block_ptr)
        
        # Sij: (Br, Bc)
        Sij = tl.dot(Qi, tl.trans(Kj)) * softmax_score
        m_t_ij = tl.max(Sij, axis = 1)

        P_t_ij = tl.exp(Sij - m_t_ij[:, None])
        l_t_ij = tl.sum(P_t_ij, axis = 1)

        mi_new = tl.maximum(mi, m_t_ij)
        li_new = tl.exp(mi - mi_new) * li + tl.exp(m_t_ij - mi_new) * l_t_ij # TODO: issue with triton exponent, ask Yaro

        # TODO: this materializes Oi_new which also takes up sram space.
        Oi_new = (li * tl.exp(mi - mi_new))[:, None] * Oi
        Oi_new += (tl.exp(m_t_ij - mi_new))[:, None] * tl.dot(P_t_ij, Vj)
        Oi_new = Oi_new * (1 / li_new)[:, None]

        Oi = Oi_new
        li = li_new
        mi = mi_new



        # advance the block pointers
        k_block_ptr = tl.advance(k_block_ptr, (Bc, 0))
        v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))
        

    tl.store(out_block_ptr, Oi)

def flash1_fwd_wrapper(q, k, v, sm_scale):
    q_cont, k_cont, v_cont = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.zeros_like(q_cont, dtype=q.dtype)

    B, N, H, D = q_cont.shape
    l = torch.zeros((B, H, N), dtype=q.dtype, device=q.device)
    m = float("-inf") * torch.ones((B, H, N), dtype=q.dtype, device=q.device)


    # l and m will have the same strides
    lm_batch_stride, lm_heads_stride, _ = m.stride()
    batch_stride, seqlen_stride, nheads_stride, _ = q_cont.stride()


    def grid(META):
        Tr = triton.cdiv(N, META['Br'])
        return (B, H, Tr)

    attention_triton[grid](
        q_cont, k_cont, v_cont, 
        output, sm_scale,
        N, H, D,
        l, m,
        lm_batch_stride, lm_heads_stride,
        batch_stride, seqlen_stride, nheads_stride,
    )

    return output

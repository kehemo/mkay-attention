import torch
import triton
import triton.language as tl
import numpy as np
from triton.runtime import driver
import time

import math
# from alternate_forward import naive_forward, flash_torch_forward
from attn_utils import force_contiguous, compute_relative_rmse

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash2_bwd.py
"""

def naive_forward_batched_supports(Q, K, V, softmax_scale):
    """
    Q: (b, t, h, d)
    K/V: (b, s, h, d)

    returns:
    S/P: (b, h, t, s)
    O: (b, t, h, d)
    L/l/m: (b, h, t)
    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)

    m = torch.max(S, dim = -1)[0]
    l = torch.sum(torch.exp(S - m[:,:,:, None]), dim = -1)
    L = (m + torch.log(l)) / np.log(2)
    return S, P, O, L, l, m

def naive_backward_batched(Q, K, V, O, dO, L, softmax_scale):
    """
    O = (b, t, h, d)
    dO = (b, t, h, d)
    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    # O = torch.einsum("b h t s, b s h d -> b t h d", P, V)

    D = torch.einsum("b t h d, b t h d -> b h t", O, dO)

    """CHECK THIS!!"""
    dV = torch.einsum("b h t s, b t h d -> b s h d", P, dO)
    dP = torch.einsum("b t h d, b s h d -> b h t s", dO, V)
    
    t = (P * dP).sum(axis = -1)[:,:,:,None]
    dS = P * (dP - t)

    dQ = torch.einsum("b h t s, b s h d -> b t h d", dS, K) * softmax_scale
    dK = torch.einsum("b h t s, b t h d -> b s h d", dS, Q) * softmax_scale
    

    return dQ, dK, dV, D, dS, dP

def flash2_backward_torch_dkdv(Q, K, V, O, dO, L, softmax_scale):
    """
    Q = (t, d)
    K, V = (s, d)
    O = (t, d)
    dO = (t, d)
    L = (t,)
    """
    N, d = Q.shape
    Bc, Br = 16, 16

    Tc = math.ceil(N / Bc)
    Tr = math.ceil(N / Br)

    dK = torch.zeros(N, d, device = Q.device, dtype = Q.dtype)
    dV = torch.zeros(N, d, device = Q.device, dtype = Q.dtype)
    D = (dO * O).sum(axis = 1)

    for j in range(Tc):
        kv_start = j * Bc
        kv_end = (j + 1) * Bc
        if kv_end > N: raise Exception("Invalid range")

        Kj = K[kv_start:kv_end, :]
        Vj = V[kv_start:kv_end, :]
        dK_j = torch.zeros(Bc, d, device = Q.device, dtype = Q.dtype)
        dV_j = torch.zeros(Bc, d, device = Q.device, dtype = Q.dtype)

        for i in range(Tr):
        # for i in range(1):
            q_start = i * Br
            q_end = (i + 1) * Br
            if q_end > N: raise Exception("Invalid range")

            Qi = Q[q_start:q_end, :]
            Oi = O[q_start:q_end, :]
            dOi = dO[q_start:q_end, :]

            Di = D[q_start:q_end]
            Li = L[q_start:q_end]

            Sij = Qi @ Kj.T * softmax_scale
            Pij = torch.exp2(Sij / np.log(2) - Li[:, None])

            dV_j = dV_j + Pij.T @ dOi
            dPij = dOi @ Vj.T

            # Di = (dOi * Oi).sum(axis = 1)
            dSij = Pij * (dPij - Di[:, None])

            dK_j = dK_j + dSij.T @ Qi * softmax_scale

        dK[kv_start:kv_end, :] = dK_j
        dV[kv_start:kv_end, :] = dV_j
    
    return dK, dV

def flash2_backward_torch_dq(Q, K, V, O, dO, L, softmax_scale):

    N, d = Q.shape
    Bc, Br = 16, 16

    Tc = math.ceil(N / Bc)
    Tr = math.ceil(N / Br)

    dQ = torch.zeros(N, d, device = Q.device, dtype = Q.dtype)
    dS = torch.zeros(N, N, device = Q.device, dtype = Q.dtype)
    D = (dO * O).sum(axis = 1)


    # for i in range(Tr):
    for i in range(Tr):
        q_start = i * Br
        q_end = (i + 1) * Br
        if q_end > N: raise Exception("Invalid range")

        Qi = Q[q_start:q_end, :]
        dOi = dO[q_start:q_end, :]
        Di = D[q_start:q_end]
        dQi = torch.zeros(Br, d, device = Q.device, dtype = Q.dtype)

        Li = L[q_start:q_end]
        for j in range(Tc):
            kv_start = j * Bc
            kv_end = (j + 1) * Bc
            if kv_end > N: raise Exception("Invalid range")

            Kj = K[kv_start:kv_end, :]
            Vj = V[kv_start:kv_end, :]

            Sij = Qi @ Kj.T * softmax_scale
            Pij = torch.exp2(Sij / np.log(2) - Li[:, None])


            dP_ij = dOi @ Vj.T

            dSij = Pij * (dP_ij - Di[:, None])
            dS[q_start:q_end, kv_start:kv_end] = dSij

            dQi += dSij @ Kj * softmax_scale

        dQ[q_start:q_end, :] = dQi
    
    return dQ, dS

@triton.jit
def flash2_bwd_prep(
            O_ptr, dO_ptr, D_ptr, 
            batch_sz, nheads, seqlen, hdim: tl.constexpr,
            D_batch_stride, D_heads_stride,
            O_batch_stride, O_heads_stride, O_seqlen_stride,
            dO_batch_stride, dO_heads_stride, dO_seqlen_stride,
            TOKEN_BLOCK: tl.constexpr):
    """
    O_ptr: (batch_sz, seqlen, nheads, hdim)
    dO_ptr: (batch_sz, seqlen, nheads, hdim)
    D_ptr: (batch_sz, nheads, seqlen)
    """
    token_block_id = tl.program_id(axis = 0)
    batch_id = tl.program_id(axis = 1)
    head_id = tl.program_id(axis = 2)

    token_block_offset = token_block_id * TOKEN_BLOCK
    BH_offset = batch_id * O_batch_stride + head_id * O_heads_stride

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + BH_offset,
        shape = (seqlen, hdim),
        strides = (O_seqlen_stride, 1),
        offsets = (token_block_offset, 0),
        block_shape = (TOKEN_BLOCK, hdim),
        order = (1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + BH_offset,
        shape = (seqlen, hdim),
        strides = (dO_seqlen_stride, 1),
        offsets = (token_block_offset, 0),
        block_shape = (TOKEN_BLOCK, hdim),
        order = (1, 0)
    )


    O = tl.load(O_block_ptr)
    dO = tl.load(dO_block_ptr)

    D = tl.sum(O * dO, axis = 1)
    D_ptrs = D_ptr + batch_id * D_batch_stride + head_id * D_heads_stride + token_block_offset + tl.arange(0, TOKEN_BLOCK)
    tl.store(D_ptrs, D)

@triton.jit
def flash2_bwd_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, softmax_scale, ln2,
    dQ_ptr, D_ptr,
    batch_sz, nheads, seqlen, hdim: tl.constexpr,
    dO_batch_stride, dO_seqlen_stride, dO_heads_stride,
    QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
    LD_batch_stride, LD_heads_stride,
    Bc: tl.constexpr, Br: tl.constexpr
):
    Tr_i = tl.program_id(axis = 0)
    batch_id = tl.program_id(axis = 1)
    head_id = tl.program_id(axis = 2)

    i_offset = Tr_i * Br

    QKV_BH_offset = batch_id * QKV_batch_stride + head_id * QKV_heads_stride
    dO_BH_offset = batch_id * dO_batch_stride + head_id * dO_heads_stride

    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (i_offset, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + dO_BH_offset,
        shape = (seqlen, hdim),
        strides = (dO_seqlen_stride, 1),
        offsets = (i_offset, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )
    Li_ptrs = L_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)
    Di_ptrs = D_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)


    Kt_block_ptr = tl.make_block_ptr(
        base = K_ptr + QKV_BH_offset,
        shape = (hdim, seqlen),
        strides = (1, QKV_seqlen_stride),
        offsets = (0, 0),
        block_shape = (hdim, Bc),
        order = (1, 0)
    )
    Vt_block_ptr = tl.make_block_ptr(
        base = V_ptr + QKV_BH_offset,
        shape = (hdim, seqlen),
        strides = (1, QKV_seqlen_stride),
        offsets = (0, 0),
        block_shape = (hdim, Bc),
        order = (1, 0)
    )


    dOi = tl.load(dO_block_ptr)
    Qi = tl.load(Q_block_ptr)
    dQi = tl.zeros((Br, hdim), dtype = Qi.dtype)
    Li = tl.load(Li_ptrs)
    Di = tl.load(Di_ptrs)

    Tc = tl.cdiv(seqlen, Bc)
    for j in range(Tc):
        KjT = tl.load(Kt_block_ptr)
        VjT = tl.load(Vt_block_ptr)

        Sij = tl.dot(Qi, KjT) * softmax_scale
        Pij = tl.exp2(Sij / ln2 - Li[:, None])


        dPij = tl.dot(dOi, VjT, allow_tf32=False)
        dSij = Pij * (dPij - Di[:, None])

        Kj_block_ptr = tl.make_block_ptr(
            base = K_ptr + QKV_BH_offset,
            shape = (seqlen, hdim),
            strides = (QKV_seqlen_stride, 1),
            offsets = (j * Bc, 0),
            block_shape = (Bc, hdim),
            order = (1, 0)
        )
        Kj = tl.load(Kj_block_ptr)


        dQi += (tl.dot(dSij, Kj) * softmax_scale)


        Kt_block_ptr = tl.advance(Kt_block_ptr, (0, Bc))
        Vt_block_ptr = tl.advance(Vt_block_ptr, (0, Bc))


    dQ_block_ptr = tl.make_block_ptr(
        base = dQ_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (i_offset, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    tl.store(dQ_block_ptr, dQi)

"""
@triton.jit
def flash2_bwd_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    softmax_scale, ln2,
    batch_sz, nheads, seqlen, hdim: tl.constexpr,
    dO_batch_stride, dO_seqlen_stride, dO_heads_stride,
    QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
    LD_batch_stride, LD_heads_stride,
    Bc: tl.constexpr, Br: tl.constexpr
):
"""

@triton.jit
def flash2_bwd_dkdv(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    dK_ptr, dV_ptr, D_ptr, L_ptr,
    softmax_scale, ln2,
    batch_sz, nheads, seqlen, hdim: tl.constexpr,
    QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
    LD_batch_stride, LD_heads_stride,
    Bc: tl.constexpr, Br: tl.constexpr
):
    Tc_j = tl.program_id(axis = 0)
    batch_id = tl.program_id(axis = 1)
    head_id = tl.program_id(axis = 2)

    Tc_block_offset = Tc_j * Bc
    QKV_BH_offset = batch_id * QKV_batch_stride + head_id * QKV_heads_stride

    Kt_block_ptr = tl.make_block_ptr(
        base = K_ptr + QKV_BH_offset,
        shape = (hdim, seqlen),
        strides = (1, QKV_seqlen_stride),
        offsets = (0, Tc_block_offset),
        block_shape = (hdim, Bc),
        order = (1, 0)
    )
    Vt_block_ptr = tl.make_block_ptr(
        base = V_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (1, QKV_seqlen_stride),
        offsets = (0, Tc_block_offset),
        block_shape = (hdim, Bc),
        order = (1, 0)
    )

    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    KjT = tl.load(Kt_block_ptr)
    VjT = tl.load(Vt_block_ptr)

    dKj = tl.zeros((Bc, hdim), dtype = KjT.dtype)
    dVj = tl.zeros((Bc, hdim), dtype = VjT.dtype)

    Tr = tl.cdiv(seqlen, Br)
    for i in range(Tr):
    # for i in range(1):
        Qi = tl.load(Q_block_ptr)
        dOi = tl.load(dO_block_ptr)

        i_offset = i * Br
        Li_ptrs = L_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)
        Di_ptrs = D_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)
        Li = tl.load(Li_ptrs)
        Di = tl.load(Di_ptrs)

        Sij = tl.dot(Qi, KjT) * softmax_scale
        Pij = tl.exp2(Sij / ln2 - Li[:, None])

        dVj += tl.dot(Pij.T, dOi)

        dPij = tl.dot(dOi, VjT)
        dSij = Pij * (dPij - Di[:, None])
        dKj += (tl.dot(dSij.T, Qi) * softmax_scale)

        Q_block_ptr = tl.advance(Q_block_ptr, (Br, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Br, 0))


        
    dK_block_ptr = tl.make_block_ptr(
        base = dK_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (Tc_block_offset, 0),
        block_shape = (Bc, hdim),
        order = (1, 0)
    )
    dV_block_ptr = tl.make_block_ptr(
        base = dV_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (Tc_block_offset, 0),
        block_shape = (Bc, hdim),
        order = (1, 0)
    )

    tl.store(dK_block_ptr, dKj)
    tl.store(dV_block_ptr, dVj)



def flash2_bwd_wrapper(Q, K, V, O, dO, L, softmax_scale):
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # don't know if O/dO are contiguous.
    O = O.contiguous()
    dO = dO.contiguous()

    qkv_dtype, qkv_device = Q.dtype, Q.device


    batch_sz, seqlen, nheads, hdim = Q.shape
    l = torch.zeros((batch_sz, nheads, seqlen), dtype=qkv_dtype, device=qkv_device)
    m = torch.ones((batch_sz, nheads, seqlen), dtype=qkv_dtype, device=qkv_device) * float("-inf")

    # l and m will have the same strides
    lm_batch_stride, lm_heads_stride, _ = m.stride()
    batch_stride, seqlen_stride, nheads_stride, _ = Q.stride()

    D = torch.empty((batch_sz, nheads, seqlen), device=qkv_device, dtype=qkv_dtype)

    # Q, K, V, O, dO, dQ, dK,dV will have the same strides assuming they are contiguous!
    O_batch_stride, O_seqlen_stride, O_heads_stride, _ = O.stride()
    dO_batch_stride, dO_seqlen_stride, dO_heads_stride, _ = dO.stride()
    D_batch_stride, D_heads_stride, _ = D.stride()


    # TODO: add in the prep call
    def prep_grid(META):
        assert seqlen % META['TOKEN_BLOCK'] == 0, "seqlen must be divisible by TOKEN_BLOCK, since we don't use a mask" # actually don't think this is necessary anymore?
        return (triton.cdiv(seqlen, META['TOKEN_BLOCK']), batch_sz, nheads)
    print(O.device, dO.device, D.device)
    flash2_bwd_prep[prep_grid](
        O, dO, D,
        batch_sz, nheads, seqlen, hdim,
        D_batch_stride, D_heads_stride,
        O_batch_stride, O_heads_stride, O_seqlen_stride,
        dO_batch_stride, dO_heads_stride, dO_seqlen_stride,
        TOKEN_BLOCK = 32
    )


    # dQ = torch.empty_like(Q, device = qkv_device, dtype = qkv_dtype)
    dQ = torch.ones_like(Q, device = qkv_device, dtype = qkv_dtype) * -1
    print(f"dQ stride: {dQ.stride()}")
    print("QKV strides: ", Q.stride())
    print("O strides: ", O.stride())
    print("dO strides: ", dO.stride())

    Bc, Br = 16, 16

    QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride, _ = Q.stride()
    LD_batch_stride, LD_heads_stride, _ = L.stride()
    
    dq_grid = lambda META: ((triton.cdiv(seqlen, Br)), batch_sz, nheads)
    flash2_bwd_dq[dq_grid](
        Q, K, V, O, dO, L, softmax_scale, np.log(2),
        dQ, D,
        batch_sz, nheads, seqlen, hdim,
        dO_batch_stride, dO_seqlen_stride, dO_heads_stride,
        QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
        LD_batch_stride, LD_heads_stride,
        Bc = Bc, Br = Br
    )


    # def flash2_bwd_dkdv(
    #     Q_ptr, K_ptr, V_ptr, dO_ptr,
    #     dK_ptr, dV_ptr, D_ptr, L_ptr,
    #     softmax_scale, ln2,
    #     batch_sz, nheads, seqlen, hdim: tl.constexpr,
    #     QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
    #     LD_batch_stride, LD_heads_stride,
    #     Bc: tl.constexpr, Br: tl.constexpr
    # ):

    dK = torch.empty_like(K, device = qkv_device, dtype = qkv_dtype)
    dV = torch.empty_like(V, device = qkv_device, dtype = qkv_dtype)
    dkdv_grid = lambda META: ((triton.cdiv(seqlen, Bc)), batch_sz, nheads)
    flash2_bwd_dkdv[dkdv_grid](
        Q, K, V, dO,
        dK, dV, D, L,
        softmax_scale, np.log(2),
        batch_sz, nheads, seqlen, hdim,
        QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
        LD_batch_stride, LD_heads_stride,
        Bc = Bc, Br = Br
    )

    return D, dQ, dK, dV





def flash2_triton_bwd_eval(qkv):
    q, k, v = qkv.unbind(dim=2)
    B, N, H, D = q.shape

    sm_scale = 1 / math.sqrt(D)
    S_naive_fwd, P_naive_fwd, O_naive_fwd, L_naive_fwd, l_naive_fwd, m_naive_fwd = naive_forward_batched_supports(q, k, v, sm_scale)
    dO_dummy = torch.randn_like(O_naive_fwd, device = qkv.device, dtype = qkv.dtype)



    dQ_ground, dK_ground, dV_ground, D_ground, dS_ground, dP_ground = naive_backward_batched(q, k, v, O_naive_fwd, dO_dummy, L_naive_fwd, sm_scale)
    dQ_ftorch_spec, dS_ftorch_spec = flash2_backward_torch_dq(q[0,:,0,:], k[0,:,0,:], v[0,:,0,:], O_naive_fwd[0,:,0,:], dO_dummy[0,:,0,:], L_naive_fwd[0,0,:], sm_scale)
    dK_ftorch_spec, dV_ftorch_spec = flash2_backward_torch_dkdv(q[0,:,0,:], k[0,:,0,:], v[0,:,0,:], O_naive_fwd[0,:,0,:], dO_dummy[0,:,0,:], L_naive_fwd[0,0,:], sm_scale)

    print("torch/torch_dQ RMSE: ", compute_relative_rmse(dQ_ftorch_spec, dQ_ground[0,:,0,:]))
    print("torch/torch_dK RMSE: ", compute_relative_rmse(dK_ftorch_spec, dK_ground[0,:,0,:]))
    print("torch/torch_dV RMSE: ", compute_relative_rmse(dV_ftorch_spec, dV_ground[0,:,0,:]))    


    """TIME: TRITON"""

    D_flash2, dQ_flash2, dK_flash2, dV_flash2 = flash2_bwd_wrapper(q, k, v, O_naive_fwd, dO_dummy, L_naive_fwd, sm_scale)

    print("triton / torch D RMSE: ", compute_relative_rmse(D_flash2, D_ground))
    print("triton / naive_dQ RMSE: ", compute_relative_rmse(dQ_flash2, dQ_ground))
    print("triton / naive_dK RMSE: ", compute_relative_rmse(dK_flash2, dK_ground))
    print("triton / naive_dV RMSE: ", compute_relative_rmse(dV_flash2, dV_ground))

    # print("triton / torch dK ratio")
    # print(dK_flash2[0,:,0,:] / dK_ftorch_spec)

    # print("triton / torch dV ratio")
    # print(dV_flash2[0,:,0,:] / dV_ftorch_spec)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    print(f"device stats")
    print(f"NUM_SM {NUM_SM}, NUM_REGS {NUM_REGS}, SIZE_SMEM {SIZE_SMEM}, WARP_SIZE {WARP_SIZE}")
    # output: NUM_SM 84, NUM_REGS 65536, SIZE_SMEM 101376, WARP_SIZE 32

    batch_size = 8
    seq_len = 256
    n_heads = 32
    head_dim = 64
    
    qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device = 'cuda', dtype=torch.float32)
    flash2_triton_bwd_eval(qkv)
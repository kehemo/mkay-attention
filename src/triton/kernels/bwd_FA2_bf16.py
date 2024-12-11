import torch
import triton
import triton.language as tl
import numpy as np
import time

import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash2_bwd.py
"""

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

def get_cuda_autotune_config(prep = False):
    num_stages_list = [3]
    Br_list = [64]
    Bc_list = [32]
    configs = []
    for num_stages in num_stages_list:
            for Br in Br_list:
                if prep:
                    configs.append(triton.Config({'Br': Br}, num_stages=num_stages))
                    continue
                for Bc in Bc_list:
                    configs.append(triton.Config({'Br': Br, 'Bc': Bc}, num_stages=num_stages))
    return configs
@triton.autotune(
    configs=get_cuda_autotune_config(prep = True),
    key=['seqlen', 'hdim'],
)
@triton.jit
def flash2_bwd_prep(
            O_ptr, dO_ptr, D_ptr, 
            batch_sz, nheads, seqlen, hdim: tl.constexpr,
            D_batch_stride, D_heads_stride,
            O_batch_stride, O_heads_stride, O_seqlen_stride,
            dO_batch_stride, dO_heads_stride, dO_seqlen_stride,
            Br: tl.constexpr):
    """
    O_ptr: (batch_sz, seqlen, nheads, hdim)
    dO_ptr: (batch_sz, seqlen, nheads, hdim)
    D_ptr: (batch_sz, nheads, seqlen)
    """
    Tr_i = tl.program_id(axis = 0)
    batch_id = tl.program_id(axis = 1)
    head_id = tl.program_id(axis = 2)

    i_offset = Tr_i * Br
    BH_offset = batch_id * O_batch_stride + head_id * O_heads_stride

    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + BH_offset,
        shape = (seqlen, hdim),
        strides = (O_seqlen_stride, 1),
        offsets = (i_offset, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + BH_offset,
        shape = (seqlen, hdim),
        strides = (dO_seqlen_stride, 1),
        offsets = (i_offset, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )


    O = tl.load(O_block_ptr)
    dO = tl.load(dO_block_ptr)

    D = tl.sum(O * dO, axis = 1)
    D_ptrs = D_ptr + batch_id * D_batch_stride + head_id * D_heads_stride + i_offset + tl.arange(0, Br)
    tl.store(D_ptrs, D)

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['seqlen', 'hdim'],
)
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
        Kj = tl.load(Kj_block_ptr).to(tl.float32)

        dQi += (tl.dot(dSij, Kj) * softmax_scale)
        dQi = dQi.to(Qi.dtype)


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

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['seqlen', 'hdim'],
)
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
        order = (0, 1)
    )
    Vt_block_ptr = tl.make_block_ptr(
        base = V_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (1, QKV_seqlen_stride),
        offsets = (0, Tc_block_offset),
        block_shape = (hdim, Bc),
        order = (0, 1)
    )

    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Br, hdim),
        order = (0, 1)
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + QKV_BH_offset,
        shape = (seqlen, hdim),
        strides = (QKV_seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Br, hdim),
        order = (0, 1)
    )

    KjT = tl.load(Kt_block_ptr)
    VjT = tl.load(Vt_block_ptr)

    dKj = tl.zeros((Bc, hdim), dtype = KjT.dtype)
    dVj = tl.zeros((Bc, hdim), dtype = VjT.dtype)

    Tr = tl.cdiv(seqlen, Br)
    for i in range(Tr):
        Qi = tl.load(Q_block_ptr)
        dOi = tl.load(dO_block_ptr)

        i_offset = i * Br
        Li_ptrs = L_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)
        Di_ptrs = D_ptr + batch_id * LD_batch_stride + head_id * LD_heads_stride + i_offset + tl.arange(0, Br)
        Li = tl.load(Li_ptrs)
        Di = tl.load(Di_ptrs)

        Sij = tl.dot(Qi, KjT) * softmax_scale
        Pij = tl.exp2(Sij / ln2 - Li[:, None])
        Pij = Pij.to(tl.bfloat16)
        dVj += tl.dot(Pij.T, dOi)
        dVj = dVj.to(tl.bfloat16)

        dPij = tl.dot(dOi, VjT)
        dSij = Pij * (dPij - Di[:, None])
        dSij = dSij.to(tl.bfloat16)
        dKj += (tl.dot(dSij.T, Qi) * softmax_scale)
        dKj = dKj.to(tl.bfloat16)

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


    D = torch.empty((batch_sz, nheads, seqlen), device=qkv_device, dtype=qkv_dtype)

    # Q, K, V, O, dO, dQ, dK,dV will have the same strides assuming they are contiguous!
    O_batch_stride, O_seqlen_stride, O_heads_stride, _ = O.stride()
    dO_batch_stride, dO_seqlen_stride, dO_heads_stride, _ = dO.stride()
    D_batch_stride, D_heads_stride, _ = D.stride()



    # Bc, Br = 16, 16
    # assert seqlen % Br == 0, "seqlen must be divisible by Br, at least it used to be (although maybe we're good on this now that we use block_ptrs?)"


    prep_grid = lambda META: ((triton.cdiv(seqlen, META["Br"])), batch_sz, nheads)
    flash2_bwd_prep[prep_grid](
        O, dO, D,
        batch_sz, nheads, seqlen, hdim,
        D_batch_stride, D_heads_stride,
        O_batch_stride, O_heads_stride, O_seqlen_stride,
        dO_batch_stride, dO_heads_stride, dO_seqlen_stride,
    )
    # sync
    # torch.cuda.synchronize()

    """Compute dQ"""
    dQ = torch.ones_like(Q, device = qkv_device, dtype = qkv_dtype) * -1
    QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride, _ = Q.stride()
    LD_batch_stride, LD_heads_stride, _ = L.stride()
    
    dq_grid = lambda META: ((triton.cdiv(seqlen, META["Br"])), batch_sz, nheads)
    flash2_bwd_dq[dq_grid](
        Q, K, V, O, dO, L, softmax_scale, np.log(2),
        dQ, D,
        batch_sz, nheads, seqlen, hdim,
        dO_batch_stride, dO_seqlen_stride, dO_heads_stride,
        QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
        LD_batch_stride, LD_heads_stride,
    )

    """Compute dK, dV"""
    dK = torch.empty_like(K, device = qkv_device, dtype = qkv_dtype)
    dV = torch.empty_like(V, device = qkv_device, dtype = qkv_dtype)
    dkdv_grid = lambda META: ((triton.cdiv(seqlen, META["Bc"])), batch_sz, nheads)
    flash2_bwd_dkdv[dkdv_grid](
        Q, K, V, dO,
        dK, dV, D, L,
        softmax_scale, np.log(2),
        batch_sz, nheads, seqlen, hdim,
        QKV_batch_stride, QKV_seqlen_stride, QKV_heads_stride,
        LD_batch_stride, LD_heads_stride,
    )

    return D, dQ, dK, dV
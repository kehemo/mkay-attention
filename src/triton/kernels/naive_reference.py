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

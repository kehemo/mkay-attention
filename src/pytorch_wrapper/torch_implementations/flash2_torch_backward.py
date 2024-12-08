import torch
import flash_torch_forward
import numpy as np
import math

# Ensure your q, k, v tensors are initialized
sram_size_bytes = 1000  # Test value, real is 100K, but this lets us split 128 seqlen into T_r, B_r
N, d = 64, 32  # Example dimensions
M = (4 * d) * 4
q = torch.randn(N, d)
k = torch.randn(N, d)
v = torch.randn(N, d)
dO = torch.randn(N, d)
sm_scale = 1 / math.sqrt(d)

def naive_backward(Q, K, V, P, dO):
    """
    Q,K,V = (t/s, d)
    S = (t, s)
    P = (t, s)
    O = (t, d)
    """
    dV = torch.einsum("t s, t d -> s d", P, dO)
    dP = torch.einsum("t d, s d -> t s", dO, V)

    t = (P * dP).sum(axis = -1)[:,None]
    dS = P * (dP - t)

    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)

    dQ = torch.einsum("t s, s d -> t d", dS, K) * softmax_scale
    dK = torch.einsum("t s, t d -> s d", dS, Q) * softmax_scale

    return dQ, dK, dV, dS, dP

from flash2_torch_forward import flash2_forward, naive_forward

def flash2_backward(Q, K, V, O, dO, L, softmax_scale):
    """
    Q = (t, d)
    K, V = (s, d)
    O = (t, d)
    dO = (t, d)
    L = (t,)
    """

    Bc = math.ceil(M / (4 * d))
    Br = min(Bc, d)

    Tc = math.ceil(N  / Bc)
    Tr = math.ceil(N / Br)

    dQ = torch.zeros(N, d)
    dK = torch.zeros(N, d)
    dV = torch.zeros(N, d)
    dS = torch.zeros(N, N)
    dP = torch.zeros(N, N)

    S = torch.zeros(N, N)
    P = torch.zeros(N, N)

    for j in range(Tc):
        kv_start = j * Bc
        kv_end = (j + 1) * Bc
        if kv_end > N: raise Exception("Invalid range")

        Kj = K[kv_start:kv_end, :]
        Vj = V[kv_start:kv_end, :]
        dK_j = torch.zeros(Bc, d)
        dV_j = torch.zeros(Bc, d)

        for i in range(Tr):
            q_start = i * Br
            q_end = (i + 1) * Br
            if q_end > N: raise Exception("Invalid range")

            Qi = Q[q_start:q_end, :]
            Oi = O[q_start:q_end, :]
            dOi = dO[q_start:q_end, :]

            Li = L[q_start:q_end]

            Sij = Qi @ Kj.T * softmax_scale
            Pij = torch.exp2(Sij / np.log(2) - Li[:, None])

            S[q_start:q_end, kv_start:kv_end] = Sij
            P[q_start:q_end, kv_start:kv_end] = Pij

            dV_j = dV_j + Pij.T @ dOi
            dP_ij = dOi @ Vj.T
            dP[q_start:q_end, kv_start:kv_end] = dP_ij # optional

            Di = (dOi * Oi).sum(axis = 1)
            dSij = Pij * (dP_ij - Di[:, None])
            dS[q_start:q_end, kv_start:kv_end] = dSij # optional

            dQi = dQ[q_start:q_end, :]
            dQ[q_start:q_end, :] = dQi + dSij @ Kj * softmax_scale
            dK_j = dK_j + dSij.T @ Qi * softmax_scale

        dK[kv_start:kv_end, :] = dK_j
        dV[kv_start:kv_end, :] = dV_j
    
    return dQ, dK, dV, dS, dP, S, P

from utils import check_tensor

def check_backward(Q, K, V, dO, softmax_scale):
    naive_S, naive_P, naive_O, naive_L, naive_l, naive_m = naive_forward(Q, K, V, softmax_scale)
    naive_dQ, naive_dK, naive_dV, naive_dS, naive_dP = naive_backward(Q, K, V, naive_P, dO)
    flash2_S, flash2_O, flash2_L, flash2_l, flash2_m = flash2_forward(Q, K, V, softmax_scale)


    check_tensor("O", naive_O, flash2_O)
    check_tensor("L", naive_L, flash2_L)
    

    f_out = flash2_backward(Q, K, V, flash2_O, dO, flash2_L, softmax_scale)
    f_dQ, f_dK, f_dV, f_dS, f_dP, f_bS, f_bP = f_out

    check_tensor("backprop_S", naive_S, f_bS)  # Check that S is reconstructed correctly
    check_tensor("backprop_P", naive_P, f_bP)  # Check that P is reconstructed correctly

    check_tensor("dP", naive_dP, f_dP)
    check_tensor("dS", naive_dS, f_dS)
    check_tensor("dQ", naive_dQ, f_dQ)
    check_tensor("dK", naive_dK, f_dK)
    check_tensor("dV", naive_dV, f_dV)
    print("Backward check complete")



check_backward(q, k, v, dO, sm_scale)
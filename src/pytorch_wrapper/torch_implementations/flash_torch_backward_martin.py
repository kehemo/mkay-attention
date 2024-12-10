import torch
import flash_torch_forward
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

def naive_forward(Q, K, V, softmax_scale):
    S = torch.einsum("t d, s d -> t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # Softmax along the key dimension
    O = torch.einsum("t s, s d -> t d", P, V)
    return S, P, O

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

def check_backward(Q, K, V, dO, softmax_scale):
    def check_tensor(name, expected, actual):
        try:
            assert torch.allclose(expected, actual, rtol = 1e-5, atol = 5e-3), "Tensor wrong"
        except:
            print(f"Expected for {name}")
            print(expected)
            print("Actual")
            print(actual)

    naive_S, naive_P, naive_O = naive_forward(Q, K, V, softmax_scale)
    naive_dQ, naive_dK, naive_dV, naive_dS, naive_dP = naive_backward(Q, K, V, naive_P, dO)

    flash_S, flash_O, l, m = flash_forward(Q, K, V, softmax_scale)

    check_tensor("S", naive_S, flash_S)
    # Didn't check "P" because naive_P and flash_P are different formats, kinda
    check_tensor("O", naive_O, flash_O)

    f_out = flash_backward(Q, K, V, flash_O, dO, l, m, softmax_scale)
    f_dQ, f_dK, f_dV, f_dS, f_dP, f_bS, f_bP = f_out

    check_tensor("bS", naive_S, f_bS)  # Check the S is reconstructed correctly
    check_tensor("bP", naive_P, f_bP)  # Check tho P is reconstructed correctly

    check_tensor("dP", naive_dP, f_dP)
    check_tensor("dS", naive_dS, f_dS)
    check_tensor("dQ", naive_dQ, f_dQ)
    check_tensor("dK", naive_dK, f_dK)
    check_tensor("dV", naive_dV, f_dV)
    print("Backward check complete")

def flash_forward(Q, K, V, softmax_scale = 1.0):
    """
    Q,K,V: (N,d)
    """
    N, d = q.size()


    Bc = math.ceil(M / (4 * d))
    Br = min(Bc, d)

    Tc = math.ceil(N  / Bc)
    Tr = math.ceil(N / Br)

    O = torch.zeros(N, d)
    S = torch.zeros(N, N)
    P = torch.zeros(N, N)
    l = torch.zeros(N)
    m = torch.ones(N) * float('-inf')

    for j in range(Tc):
        kv_start = j * Bc
        kv_end = (j + 1) * Bc
        if kv_end > N: raise Exception("Invalid range")
        
        Kj = K[kv_start:kv_end, :]
        Vj = V[kv_start:kv_end, :]
        for i in range(Tr):
            q_start = i * Br
            q_end = (i + 1) * Br
            if q_end > N: raise Exception("Invalid range")
            Qi = Q[q_start:q_end, :]
            Oi = O[q_start:q_end, :]
            li = l[q_start:q_end]
            mi = m[q_start:q_end]

            # Br x Bc
            Sij = Qi @ Kj.T * softmax_scale
            S[q_start:q_end, kv_start:kv_end] = Sij # Debugging


            m_t_ij = Sij.max(dim = 1)[0] # don't care about the indices, since this returns a tuple
            P_t_ij = torch.exp(Sij - m_t_ij[:, None])
            P[q_start:q_end, kv_start:kv_end] = P_t_ij  # Debugging, didn't return
            l_t_ij = torch.sum(P_t_ij, dim = 1)

            mi_new = torch.max(m_t_ij, mi)
            # breakpoint()
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_t_ij - mi_new) * l_t_ij

            # breakpoint()
            Oi_new = (li * torch.exp(mi - mi_new))[:, None] * Oi
            Oi_new += (torch.exp(m_t_ij - mi_new))[:,None] * P_t_ij @ Vj
            Oi_new = Oi_new * (1 / li_new)[: , None]

            O[q_start:q_end, :] = Oi_new
            l[q_start:q_end] = li_new
            m[q_start:q_end] = mi_new

    return S, O, l, m


def flash_backward(Q, K, V, O, dO, l, m, softmax_scale):
    """
    Q,K,V: (N,d)
    """
    N, d = q.size()

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
        dKj = dK[kv_start:kv_end, :]
        dVj = dV[kv_start:kv_end, :]
        for i in range(Tr):
            q_start = i * Br
            q_end = (i + 1) * Br
            if q_end > N: raise Exception("Invalid range")

            # Step 10
            Qi = Q[q_start:q_end, :]
            Oi = O[q_start:q_end, :]
            dOi = dO[q_start:q_end, :]
            dQi = dQ[q_start:q_end, :]
            li = l[q_start:q_end]
            mi = m[q_start:q_end]

            # Step 11
            Sij = Qi @ Kj.T * softmax_scale
            S[q_start:q_end, kv_start:kv_end] = Sij  # optional writeback for debugging

            # Step 13 (skip 12)
            Pij = (1 / li)[:, None] * torch.exp(Sij - mi[:, None])
            P[q_start:q_end, kv_start:kv_end] = Pij  # optional writeback for debugging

            # Step 16 (skip 13, 14)
            dVj[:] = dVj + Pij.T @ dOi

            # Step 17
            dPij = dOi @ Vj.T
            dP[q_start:q_end, kv_start:kv_end] = dPij  # optional writeback for debugging

            # Step 19 (skip 18)
            Di = (dOi * Oi).sum(1)

            # Step 20
            dSij = Pij * (dPij - Di[:, None])  # Br, Bc
            dS[q_start:q_end, kv_start:kv_end] = dSij  # optional writeback for debugging

            # For these I'm writing to the outer one, but in CUDA you
            # would want to write to the local one, then only after the
            # i-loop write to the outer one.

            # Step 21
            dQi[:] = dQi + dSij @ Kj * softmax_scale

            # Step 22
            dKj[:] = dKj + dSij.T @ Qi * softmax_scale

    return dQ, dK, dV, dS, dP, S, P # for debugging

check_backward(q, k, v, dO, sm_scale)
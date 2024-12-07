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

    return dQ, dK, dV

def check_backward(Q, K, V, dO, softmax_scale):
    def check_tensor(expected, actual):
        assert torch.allclose(expected, actual, rtol = 1e-5, atol = 5e-3), "Tensor wrong"

    naive_S, naive_P, naive_O = naive_forward(Q, K, V, softmax_scale)
    naive_dQ, naive_dK, naive_dV = naive_backward(Q, K, V, naive_P, dO)

    flash_S, flash_O, l, m = flash_forward(Q, K, V, softmax_scale)

    check_tensor(naive_S, flash_S)
    check_tensor(naive_O, flash_O)

    flash_out = flash_backward(Q, K, V, dO, l, m, softmax_scale)
    flash_dQ, flash_dK, flash_dV = flash_out

    check_tensor(naive_dQ, flash_dQ)
    check_tensor(naive_dK, flash_dK)
    check_tensor(naive_dV, flash_dV)
    print("All pass")

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
            S[q_start:q_end, kv_start:kv_end] = Sij


            m_t_ij = Sij.max(dim = 1)[0] # don't care about the indices, since this returns a tuple
            P_t_ij = torch.exp(Sij - m_t_ij[:, None])
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


def flash_backward(Q, K, V, dO, l, m, softmax_scale):
    raise NotImplementedError

check_backward(q, k, v, dO, sm_scale)
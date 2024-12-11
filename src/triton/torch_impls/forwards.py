import torch
import math


def naive_forward(Q, K, V, softmax_scale):
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)
    return S, P, O

def flash_torch_forward(Q, K, V, softmax_scale = 1.0):
    """
    Q,K,V: (B,N,H,d)
    """
    Q = Q[0, :, 0, :].to('cpu')
    K = K[0, :, 0, :].to('cpu')
    V = V[0, :, 0, :].to('cpu')
    N, d = Q.size()


    # Bc = math.ceil(M / (4 * d))
    # Br = min(Bc, d)

    Bc = 16
    Br = 16

    Tc = math.ceil(N  / Bc)
    Tr = math.ceil(N / Br)

    O = torch.zeros(N, d)
    S = torch.zeros(N, N)
    l = torch.zeros(N)
    m = torch.ones(N) * float('-inf')

    for j in range(0,Tc):
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
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_t_ij - mi_new) * l_t_ij

            Oi_new = (li * torch.exp(mi - mi_new))[:, None] * Oi
            Oi_new += (torch.exp(m_t_ij - mi_new))[:,None] * P_t_ij @ Vj
            Oi_new = Oi_new * (1 / li_new)[: , None]

            O[q_start:q_end, :] = Oi_new
            l[q_start:q_end] = li_new
            m[q_start:q_end] = mi_new






    return S, O
import torch
import math

# Ensure your q, k, v tensors are initialized
sram_size_bytes = 1000  # Test value, real is 100K, but this lets us split 128 seqlen into T_r, B_r
N, d = 64, 32  # Example dimensions
M = (4 * d) * 4
q = torch.randn(N, d)
k = torch.randn(N, d)
v = torch.randn(N, d)

def naive_forward(Q, K, V, softmax_scale):
    S = torch.einsum("t d, s d -> t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # Softmax along the key dimension
    O = torch.einsum("t s, s d -> t d", P, V)
    return S, P, O



def flash_forward(Q, K, V, softmax_scale = 1.0):
    """
    Q,K,V: (N,d)
    """
    N, d = Q.size()


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





    return S, O


sm_scale = 1 / math.sqrt(d)
S_naive, P_naive, O_naive = naive_forward(q, k, v, sm_scale)
S_flash, O_flash = flash_forward(q, k, v, sm_scale)


# print("S_naive")
# print(S_naive)
# print("S_flash")
# print(S_flash)
assert torch.allclose(S_naive, S_flash, rtol = 1e-5, atol = 5e-3), "S was incorrect"


# print("O_naive")
# print(O_naive)
# print("O_flash")
# print(O_flash)
assert torch.allclose(O_naive, O_flash, rtol = 1e-5, atol = 5e-3), "O was incorrect"
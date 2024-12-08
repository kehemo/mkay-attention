import torch
import numpy as np
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

    m = torch.max(S, dim = 1)[0]
    l = torch.sum(torch.exp(S - m[:, None]), dim = 1)
    L = (m + torch.log(l)) / np.log(2)
    return S, P, O, L, l, m


def flash2_forward(Q, K, V, softmax_scale = 1.0):
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
    L = torch.empty(N)
    l = torch.zeros(N)
    m = torch.ones(N) * float('-inf')


    for i in range(Tr):
        q_start = i * Br
        q_end = (i + 1) * Br
        if q_end > N: raise Exception("Invalid range")
        Qi = Q[q_start:q_end, :]
        Oi = O[q_start:q_end, :]
        li = torch.zeros(Br)
        mi = torch.ones(Br) * float('-inf')

        for j in range(Tc):
            kv_start = j * Bc
            kv_end = (j + 1) * Bc
            if kv_end > N: raise Exception("Invalid range")
            
            Kj = K[kv_start:kv_end, :]
            Vj = V[kv_start:kv_end, :]


            # Br x Bc
            Sij = Qi @ Kj.T * softmax_scale * (1 / np.log(2))
            S[q_start:q_end, kv_start:kv_end] = Sij


            mi_new = torch.max(Sij.max(dim = 1)[0], mi)
            P_t_ij = torch.exp2((Sij - mi_new[:, None]))

            li = torch.exp2(mi - mi_new) * li + torch.sum(P_t_ij, dim = 1)
            Oi = torch.exp2(mi - mi_new)[:, None] * Oi + P_t_ij @ Vj
            mi = mi_new


        Oi_norm = (1 / li)[:, None] * Oi
        Li = mi + torch.log2(li)

        O[q_start:q_end, :] = Oi_norm
        L[q_start:q_end] = Li #* np.log(2)
        l[q_start:q_end] = li
        m[q_start:q_end] = mi

    return S, O, L, l, m


if __name__ == "__main__":
    sm_scale = 1 / math.sqrt(d)
    S_naive, P_naive, O_naive, L_naive, l_naive, m_naive = naive_forward(q, k, v, sm_scale)
    S_flash, O_flash, L_flash, l_flash, m_flash = flash2_forward(q, k, v, sm_scale)

    """
    Why use exp2?
    https://github.com/triton-lang/triton/issues/2893
    exp2 is faster than exp because the special function unit inside the GPU implements exp2. When you use the exp intrinsic, it expands to a multiplication by 1/log(2) followed by the exp2 instruction. In this case, we can fold this 1/log(2) multiplier into the sm_scale multiplier so that we need only perform one multiplication per element rather than two.

    NOTE: because of the use of log2/exp2, l,m will not match up anymore
    """

    from utils import check_tensor

    # check_tensor("S", S_naive, S_flash)
    check_tensor("O", O_naive, O_flash)
    check_tensor("L", L_naive, L_flash)
    # check_tensor("l", l_naive, l_flash)
    # check_tensor("m", m_naive, m_flash)

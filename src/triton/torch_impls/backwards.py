import torch
import math



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

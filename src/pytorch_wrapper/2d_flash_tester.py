from einops import rearrange
import torch
import math

# Ensure your q, k, v tensors are initialized
sram_size_bytes = 1000  # Test value, real is 100K, but this lets us split 128 seqlen into T_r, B_r
seq_len, head_dim = 128, 32  # Example dimensions
q = torch.randn(seq_len, head_dim, requires_grad=True)
k = torch.randn(seq_len, head_dim, requires_grad=True)
v = torch.randn(seq_len, head_dim, requires_grad=True)

def name_shape(n: str, t: torch.Tensor):
    print(f"{n}: {t.shape}")

def naive_forward(Q, K, V):
    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    S = torch.einsum("t d, s d -> t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # Softmax along the key dimension
    O = torch.einsum("t s, s d -> t d", P, V)
    return S, P, O

def flash_forward_model(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    print("Running flash forward model")
    seqlen, d = Q.shape
    print(f"Parameters: t/s = {seq_len}, d={d}")
    softmax_scale = 1.0 / math.sqrt(d)

    B_c = math.ceil(sram_size_bytes / (4 * d))
    B_r = min(B_c, d)

    T_r = math.ceil(seqlen / B_r)  # Not affected by num heads or batch_size
    T_c = math.ceil(seqlen / B_c)

    assert seqlen == T_r * B_r, f"N ({seqlen}) must equal T_r * B_r ({T_r * B_r})"

    # Step 3
    Qs = rearrange(q, '(g t) d -> g t d', g=T_r, t=B_r)
    Ks = rearrange(k, '(g s) d -> g s d', g=T_c, s=B_c)
    Vs = rearrange(v, '(g s) d -> g s d', g=T_c, s=B_c)
    """
    where T is the number of blocks, and B is block size (B_r for t, B_c for s)
    Qs = (T_r, B_r, d)
    Ks = (T_c, B_c, d)
    Vs = (T_c, B_c, d)
    """

    print("(Step 3) Shapes of the block-divided QKV")
    print(f"T_r = {T_r}, T_c = {T_c}, B_r = {B_r}, B_c = {B_c}")
    name_shape("Qs", Qs)
    name_shape("Ks", Ks)
    name_shape("Vs", Vs)

    lm_shape = q.shape[:-1]
    # lm_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
    O = torch.zeros_like(Q)
    l = torch.zeros(lm_shape)
    m = torch.full(lm_shape, float('-inf'))

    print("Shapes of O, l, m")
    name_shape("O", O)
    name_shape("l", l)
    name_shape("m", m)

    # Step 4 (Incomplete)
    Os = rearrange(O, '(g t) d -> g t d', g=T_r, t=B_r)
    # need statistic generation
    ls = rearrange(l, "(g t) -> g t", g=T_r, t=B_r)
    ms = rearrange(m, "(g t) -> g t", g=T_r, t=B_r)
    """
    Os  = (T_r, B_r, d)
    ls  = (T_r, B_r)  # Reduced across d already
    ms  = (T_r, B_r)
    """
    print("(Step 4) Shapes of block-divided O, ls, ms")
    name_shape("Os", Os)
    name_shape("ls", ls)
    name_shape("ms", ms)
    # Step 5

    # Note about data layout. You may want to do this differently in CUDA. This is just
    # to get things started. You may want to e.g., change the l, m layouts.
    for j in range(T_c):
        # Select for particular group
        K_j = Ks[j, :, :]
        V_j = Vs[j, :, :]
        for i in range(T_r):
            is_first = i == 0 and j == 0
            Q_i = Qs[i, :, :]
            O_i = Os[i, :, :]
            l_i = ls[i, :]
            m_i = ms[i, :]  # Should be updated every iteration

            S_ij = softmax_scale * torch.einsum("td, sd -> ts", Q_i, K_j)  # Step 10
            m_ij = S_ij.max(-1).values  # bht1
            if is_first:
                print("Inner loop shapes")
                name_shape("S_ij", S_ij)
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = P_ij.sum(-1)

            if is_first:
                name_shape("m_ij", m_ij)
                name_shape("l_ij", l_ij)
            m_new = torch.maximum(m_ij, m_i)

            # adjust_old_factor1 = torch.diag()

            exp_mi = torch.exp(m_i - m_new)
            exp_mij= torch.exp(m_ij - m_new)
            l_new = exp_mi * l_i +  exp_mij * l_ij

            o_factor = l_new.reciprocal()
            diag_l_i = torch.diag(l_i)
            if is_first:
                name_shape("o_factor", o_factor)
                name_shape("l_i", l_i)
                name_shape("diag_l_i", diag_l_i)
                name_shape("exp_mi", exp_mi)
                name_shape("O_i", O_i)
            o_term1 = l_i.unsqueeze(1) * (exp_mi.unsqueeze(1) * O_i)
            o_term2 = exp_mij.unsqueeze(1) * P_ij @ V_j
            if is_first:
                name_shape("o_term1", o_term1)
                name_shape("o_term2", o_term2)
            O_i[:] = o_factor.unsqueeze(1) * (o_term1 + o_term2)
            m_i[:] = m_new
            l_i[:] = l_new
            # This stuff is actually supposed to happen in different blocks, but we're
            # doing all of them simultaneously here.
            # S_ = softmax_scale * torch.einsum("b, ", Q, K)
    # print(f"l: {l}")
    # print(f"m: {m}")

    return O, l, m

"""
To properly implement the backward pass, we need access to the l, m statistics from the
FlashAttention forward pass. We generate these separately for testing, though these should
correspond to the real thing.
"""
def generate_statistics(Q, K):
    """
    m is a function of S and is the max across the `d` dimension.
    """
    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    S = torch.einsum("t d, s d -> t s", Q, K) * softmax_scale
    # max over d of bthd
    m = S.max(-1, keepdim=True).values  # t 1
    P_ = torch.exp(S - m) # t s
    l = P_.sum(-1, keepdim=True)  # t 1

    print(f"Shape of m is {m.shape} and l is {l.shape}")
    return l, m

# def check_statistics(Q, K, V):
#     l, m = generate_statistics(Q, K)
#     S, P, O = naive_forward(Q, K, V)

#     # Now, it should be that softmax(x) = f(x) / l(x), where
#     # f(x) = exp(x - m)
#     # l(x) = sum of f(x)
#     try_P =

def flash_backward_model(Q, K, V, S, P, O, dO, l, m):
    
    """
    Q,K,V = (b, t/s, h, d)
    S = (b, h, t, s)
    P = (b, h, t, s)
    O = (b, t, h, d)
    """
    seqlen, nheads, d = Q.shape
    softmax_scale = 1.0 / math.sqrt(d)

    # Stopgap: should really be generating these from the forward pass
    l, m = generate_statistics(Q, K)  # b h t 1

    B_c = math.ceil(sram_size_bytes / (4 * d))
    B_r = min(B_c, d)

    T_r = math.ceil(seqlen / B_r)  # Not affected by num heads or batch_size
    T_c = math.ceil(seqlen / B_c)

    assert seqlen == T_r * B_r, f"N ({seqlen}) must equal T_r * B_r ({T_r * B_r})"

    # Step 3
    Qs = rearrange(q, 'b (g t) h d -> b g t h d', g=T_r, t=B_r)
    Ks = rearrange(k, 'b (g s) h d -> b g s h d', g=T_c, s=B_c)
    Vs = rearrange(v, 'b (g s) h d -> b g s h d', g=T_c, s=B_c)
    """
    where T is the number of blocks, and B is block size (B_r for t, B_c for s)
    Qs = (b, T_r, B_r, h d)
    Ks = (b, T_c, B_c, h d) 
    Vs = (b, T_c, B_c, h d)
    """
    print("(Step 3) Shapes of the block-divided QKV")
    print(f"T_r = {T_r}, T_c = {T_c}, B_r = {B_r}, B_c = {B_c}")
    print(Qs.shape)
    print(Ks.shape)
    print(Vs.shape)

    # Step 4 (Incomplete)
    Os = rearrange(O, 'b (g t) h d -> b g t h d', g=T_r, t=B_r)
    dOs = rearrange(dO, 'b (g t) h d -> b g t h d', g=T_r, t=B_r)
    # need statistic generation
    ls = rearrange(l, "b h (g t) 1 -> b h g t 1", g=T_r, t=B_r)
    ms = rearrange(m, "b h (g t) 1 -> b h g t 1", g=T_r, t=B_r)
    """
    Os  = (b, T_r, B_r, h, d)
    ls  = (b, T_r, B_r, h, 1)  # Reduced across d already
    ms  = (b, T_r, B_r, h, 1)
    """
    print("(Step 4) Shapes of block-divided O, dOs, ls, ms")
    print(Os.shape)
    print(dOs.shape)
    print(ls.shape)
    print(ms.shape)
    # Step 5
    dQs = torch.zeros_like(Qs)
    dKs = torch.zeros_like(Ks)
    dVs = torch.zeros_like(Vs)


    for j in range(T_c):
        dKw = torch.zeros_like(Ks) # I don't know convention for the swiggle on top
        dVw = torch.zeros_like(Vs)
        for j in range(T_r):
            # This stuff is actually supposed to happen in different blocks, but we're
            # doing all of them simultaneously here.
            S_ = softmax_scale * torch.einsum("b, ", Q, K)

    raise NotImplementedError
    dV = torch.einsum("b h t s, b t h d -> b s h d", P, dO)
    dP = torch.einsum("b t h d, b s h d -> b h t s", dO, V)
     
    t = (P * dP).sum(axis = -1)[:,:,:,None]
    dS = P * (dP - t)

    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    
    dQ = torch.einsum("b h t s, b s h d -> b t h d", dS, K) * softmax_scale
    dK = torch.einsum("b h t s, b t h d -> b s h d", dS, Q) * softmax_scale
    
    try:
        # i have no idea why the tolerances have to be this high....
        assert torch.allclose(dV, V.grad, rtol = 1e-5, atol = 5e-3), "dV was incorrect"
        assert torch.allclose(dQ, Q.grad, rtol = 1e-5, atol = 5e-3), "dQ was incorrect"
        assert torch.allclose(dK, K.grad, rtol = 1e-5, atol = 5e-3), "dK was incorrect"
    except:
        print("Failed")
        V_dev = (dV / V.grad - 1).abs().max()
        Q_dev = (dQ / Q.grad - 1).abs().max()
        K_dev = (dK / K.grad - 1).abs().max()
        print(V_dev, Q_dev, K_dev)

    return dQ, dK, dV

def check_statistics(q, k, v):
    l_stat, m_stat = generate_statistics(q, k)
    O, l, m = flash_forward_model(q, k, v)
    # name_shape("m_stat", m_stat)
    # name_shape("m", m)
    # name_shape("l_stat", l_stat)
    # name_shape("l", l)
    if torch.allclose(m, m_stat) and torch.allclose(l, l_stat):
        print("Statistics from flash model and statistics function match.")

# TODO: Consider finish implementing the forward model. But probably not necessary since we've
# checked the statistics separately.
def check_forward_pass(q, k, v):
    O, l, m = flash_forward_model(q, k, v)
    S, P, ref_O = naive_forward(q, k, v)

    try:
        # i have no idea why the tolerances have to be this high....
        assert torch.allclose(ref_O, O, atol = 5e-4), "O was incorrect"
        print("O was correct")
    except:
        print("Failed")
        O_dev = (O / ref_O - 1).abs().max()
        # print(O_dev)
        print(O)
        print(ref_O)

def naive_backward(Q, K, V, S, P, O, dO):
    """
    Q,K,V = (b, t/s, h, d)
    S = (b, h, t, s)
    P = (b, h, t, s)
    O = (b, t, h, d)
    """

    dV = torch.einsum("b h t s, b t h d -> b s h d", P, dO)
    dP = torch.einsum("b t h d, b s h d -> b h t s", dO, V)

    t = (P * dP).sum(axis = -1)[:,:,:,None]
    dS = P * (dP - t)

    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)

    dQ = torch.einsum("b h t s, b s h d -> b t h d", dS, K) * softmax_scale
    dK = torch.einsum("b h t s, b t h d -> b s h d", dS, Q) * softmax_scale

    return dQ, dK, dV

def check_backward_pass(q, k, v):
    # Set up the intermediate values from the forward pass
    S, P, O = naive_forward(q, k, v)
    # Scalar L computation (example: sum of all elements of O)
    O_sum = O.sum()
    L = O_sum * O_sum
    _grads = torch.autograd.grad(L, [O,S], retain_graph=True) # len 1 list

    torch_dO = _grads[0]
    torch_dS = _grads[1]
    L.backward()  # Backpropagate to compute gradients
    ref_dQ, ref_dK, ref_dV = q.grad, k.grad, v.grad

    # Example check against the naive implementation. This here is a little redundant with the
    # check inside of naive_backward.
    dQ, dK, dV = naive_backward(q, k, v, S, P, O, torch_dO)

    try:
        # i have no idea why the tolerances have to be this high....
        assert torch.allclose(ref_dQ, dQ, atol = 5e-4), "dQ was incorrect"
        assert torch.allclose(ref_dK, dK, atol = 5e-4), "dK was incorrect"
        assert torch.allclose(ref_dV, dV, atol = 5e-4), "dV was incorrect"
    except:
        print("Failed")
        Q_dev = (dQ / ref_dQ - 1).abs().max()
        K_dev = (dK / ref_dK - 1).abs().max()
        V_dev = (dV / ref_dV - 1).abs().max()
        print(Q_dev, K_dev, V_dev)

check_forward_pass(q, k, v)
# check_statistics(q,k,v)
# check_backward_pass(q, k, v)

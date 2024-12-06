from einops import rearrange
import torch
import math

# Ensure your q, k, v tensors are initialized
sram_size_bytes = 1000  # Test value, real is 100K, but this lets us split 128 seqlen into T_r, B_r
batch_size, seq_len, num_heads, head_dim = 16, 128, 4, 32  # Example dimensions
q = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)

def naive_forward(Q, K, V):
    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # Softmax along the key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)
    return S, P, O

def flash_forward_model(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    print("Running flash forward model")
    batch_size, seqlen, nheads, d = Q.shape
    print(f"Parameters: b = {batch_size}, t/s = {seq_len}, h = {nheads}, d={d}")
    softmax_scale = 1.0 / math.sqrt(d)

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

    lm_shape = q.shape[:-1]
    # lm_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
    O = torch.zeros_like(Q)
    l = torch.zeros(lm_shape).unsqueeze(3)
    m = torch.full(lm_shape, float('-inf')).unsqueeze(3)

    print("Shapes of O, l, m")
    print(O.shape)
    print(l.shape)
    print(m.shape)

    # Step 4 (Incomplete)
    Os = rearrange(O, 'b (g t) h d -> b g h t d', g=T_r, t=B_r)
    # need statistic generation
    ls = rearrange(l, "b (g t) h 1 -> b g h t 1", g=T_r, t=B_r)
    ms = rearrange(m, "b (g t) h 1 -> b g h t 1", g=T_r, t=B_r)
    """
    Os  = (b, T_r, h, B_r, d)
    ls  = (b, T_r, h, B_r, 1)  # Reduced across d already
    ms  = (b, T_r, h, B_r, 1)
    """
    print("(Step 4) Shapes of block-divided O, ls, ms")
    print(Os.shape)
    print(ls.shape)
    print(ms.shape)
    # Step 5

    # Note about data layout. You may want to do this differently in CUDA. This is just
    # to get things started. You may want to e.g., change the l, m layouts.
    for j in range(T_c):
        # Select for particular group
        K_j = Ks[:, j, :, :, :]
        V_j = Vs[:, j, :, :, :]
        for i in range(T_r):
            Q_i = Qs[:, i, :, :, :]
            O_i = Os[:, i, :, :, :]
            l_i = ls[:, i, :, :, :]
            m_i = ms[:, i, :, :, :]

            S_ij = softmax_scale * torch.einsum("bthd, bshd -> bhts", Q_i, K_j)  # Step 10
            m_ij = S_ij.max(-1, keepdim=True).values  # bht1
            if i == 0 and j == 0:
                print("Inner loop shapes")
                print(f"S_ij {S_ij.shape}")
                print(f"m {m_i.shape}")
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = P_ij.sum(-1, keepdim=True)

            if i == 0 and j == 0:
                print("m shapes")
                print(f"m_ij {m_ij.shape}, m_i {m_i.shape}")
            m_new = torch.maximum(m_ij, m_i)
            l_new = torch.exp(m_i - m_new) * torch.exp(m_ij - m_new) * l_ij

            # lol this one's a little trickier, step 15 bro
            # O_i[:] =

            m_i[:] = m_new
            l_i[:] = l_new
            # This stuff is actually supposed to happen in different blocks, but we're
            # doing all of them simultaneously here.
            # S_ = softmax_scale * torch.einsum("b, ", Q, K)

    raise NotImplementedError

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
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    # max over d of bthd
    m = S.max(-1, keepdim=True).values  # b h t 1
    P_ = torch.exp(S - m) # b h t s
    l = P_.sum(-1, keepdim=True)  # b h t 1

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
    batch_size, seqlen, nheads, d = Q.shape
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

# Compute attention output
# S, P, O = naive_forward(q, k, v)

# # Scalar L computation (example: sum of all elements of O)
# O_sum = O.sum()
# L = O_sum * O_sum


# _grads = torch.autograd.grad(L, [O,S], retain_graph=True) # len 1 list
# torch_dO = _grads[0]
# torch_dS = _grads[1]


# Compute gradients
# L.backward()  # Backpropagate to compute gradients


flash_forward_model(q, k, v)
# flash_backward_model(q,k,v, S,P,O, torch_dO, None, None)
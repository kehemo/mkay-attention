from einops import rearrange
import torch
import math

# Ensure your q, k, v tensors are initialized
sram_size_bytes = 1000  # Test value, real is 100K, but this lets us split 128 seqlen into T_r, B_r
batch_size, seq_len, num_heads, head_dim = 16, 128, 4, 32  # Example dimensions
q = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)

def name_shape(n: str, t: torch.Tensor):
    print(f"{n}: {t.shape}")

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
    name_shape("Qs", Qs)
    name_shape("Ks", Ks)
    name_shape("Vs", Vs)

    lm_shape = q.shape[:-1]
    # lm_shape = (q.shape[0], q.shape[1], q.shape[2], 1)
    O = torch.zeros_like(Q)
    l = torch.zeros(lm_shape).unsqueeze(3)
    m = torch.full(lm_shape, float('-inf')).unsqueeze(3)

    print("Shapes of O, l, m")
    name_shape("O", O)
    name_shape("l", l)
    name_shape("m", m)

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
    name_shape("Os", Os)
    name_shape("ls", ls)
    name_shape("ms", ms)
    # Step 5

    # Note about data layout. You may want to do this differently in CUDA. This is just
    # to get things started. You may want to e.g., change the l, m layouts.
    for j in range(T_c):
        # Select for particular group
        K_j = Ks[:, j, :, :, :]
        V_j = Vs[:, j, :, :, :]
        for i in range(T_r):
            is_first = i == 0 and j == 0
            Q_i = Qs[:, i, :, :, :]
            O_i = Os[:, i, :, :, :]
            l_i = ls[:, i, :, :, :]
            m_i = ms[:, i, :, :, :]  # Should be updated every iteration

            S_ij = softmax_scale * torch.einsum("bthd, bshd -> bhts", Q_i, K_j)  # Step 10
            m_ij = S_ij.max(-1, keepdim=True).values  # bht1
            if is_first:
                print("Inner loop shapes")
                name_shape("S_ij", S_ij)
            P_ij = torch.exp(S_ij - m_ij)
            l_ij = P_ij.sum(-1, keepdim=True)

            if is_first:
                name_shape("m_ij", m_ij)
                name_shape("l_ij", l_ij)
            m_new = torch.maximum(m_ij, m_i)

            # adjust_old_factor1 = torch.diag()

            exp_mi = torch.exp(m_i - m_new)
            exp_mij= torch.exp(m_ij - m_new)
            l_new = exp_mi * l_i +  exp_mij * l_ij

            # lol this one's a little trickier, step 15 bro
            # (bthd) = (bhtt)(bhtt*bthd + bht1*bhts*bshd)
            # identity = torch.eye(B_r)
            # inv_l_new_diag = torch.einsum("bhti, tu -> bhtu", torch.reciprocal(l_new), identity)
            # l_i_diag = torch.einsum("bhti, tu -> bhtu", l_i, identity)
            # if is_first:
            #     name_shape("l_diag", inv_l_new_diag)

            # Yup this is diagonal
            # view = l_new_diag[0, 0, :, :]
            # if not torch.equal(view, torch.zeros_like(view)):
            #     print(view)

            # term1 = torch.einsum()

            # o_new = torch.einsum("bhts, bshd -> bthd", P_ij, V_j)
            # o_adjustment = adjust_old_factor *
            # O_i[:] = adjust_new_factor * (o_adjustment + o_new)

            # O_i[:] = inv_l_new_diag * (l_i_diag * exp_mi * O_i + exp_mij * P_ij * V_j)
            # term1 = torch.einsum("")
            # o_term1 = exp_mi * torch.einsum(" -> bthd", l_i_diag, O_i)
            # o_term2 =  exp_mij * torch.einsum("bhts, bshd -> bthd", P_ij, V_j)
            # o_sum = o_term1 + o_term2
            # O_i[:] = torch.einsum("bhtu, bhud -> bthd", inv_l_new_diag, o_sum)
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

def check_statistics(q, k, v):
    l_stat, m_stat = generate_statistics(q, k)
    O, l, m = flash_forward_model(q, k, v)

    # Slightly different formats
    # generate_statistics has them in bthi
    # flash_forward_model has them in bhti
    m_stat = rearrange(m_stat, "b t h i -> b h t i")
    l_stat = rearrange(l_stat, "b t h i -> b h t i")
    # name_shape("m_stat", m_stat)
    # name_shape("m", m)
    # name_shape("l_stat", l_stat)
    # name_shape("l", l)
    if torch.allclose(m, m_stat) and torch.allclose(l, l_stat):
        print("Statistics from flash model and statistics function match.")

# TODO: Consider finish implementing the forward model. But probably not necessary since we've
# checked the statistics separately.
def check_forward_pass(q, k, v):
    flash_O, _, _, flash_S, flash_P = flash_forward_model(q, k, v)
    S, P, O = naive_forward(q, k, v)
    raise NotImplementedError

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

check_statistics(q,k,v)
check_backward_pass(q, k, v)
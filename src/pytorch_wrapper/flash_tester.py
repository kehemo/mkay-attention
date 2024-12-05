from einops import rearrange
import torch
import math

# Ensure your q, k, v tensors are initialized
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

def flash_backward_model(Q, K, V, S, P, O, dO):
    
    """
    Q,K,V = (b, t/s, h, d)
    S = (b, h, t, s)
    P = (b, h, t, s)
    O = (b, t, h, d)
    """
    sram_size_bytes = 1000  # Test value, real is 100K
    batch_size, seqlen, nheads, d = Q.shape

    l, m = 0, 0  # TODO need to fill in.

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
    Os = rearrange(q, 'b (g t) h d -> b g t h d', g=T_r, t=B_r)
    dOs = rearrange(q, 'b (g t) h d -> b g t h d', g=T_r, t=B_r)
    # need statistic generation
    # ls
    # ms
    """
    Os  = (b, T_r, B_r, h d)
    dOs = (b, T_r, B_r, h d)
    ls  = (b, T_r, B_r, h)  # Reduced across d already
    ms  = (b, T_r, B_r, h)
    """
    print("(Step 4) Shapes of block-divided O, dOs, ls, ms")
    print(Os.shape)
    print(dOs.shape)
    # print(ls.shape)
    # print(ms.shape)
    # Step 5
    dQs = torch.zeros_like(Qs)
    dKs = torch.zeros_like(Ks)
    dVs = torch.zeros_like(Vs)

    for j in range(T_c):
        dKw = torch.zeros_like(Ks) # I don't know convention for the swiggle on top
        dVw = torch.zeros_like(Vs)
        for j in range(T_r):
            s = torch.einsum("")
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
S, P, O = naive_forward(q, k, v)

# Scalar L computation (example: sum of all elements of O)
O_sum = O.sum()
L = O_sum * O_sum


_grads = torch.autograd.grad(L, [O,S], retain_graph=True) # len 1 list
torch_dO = _grads[0]
torch_dS = _grads[1]


# Compute gradients
L.backward()  # Backpropagate to compute gradients



flash_backward_model(q,k,v, S,P,O, torch_dO)
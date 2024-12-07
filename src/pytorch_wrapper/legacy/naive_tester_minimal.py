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



naive_backward(q,k,v, S,P,O, torch_dO)
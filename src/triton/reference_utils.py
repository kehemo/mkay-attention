import torch 
import math
import numpy as np


def naive_forward_batched_supports(Q, K, V, softmax_scale):
    """
    Q: (b, t, h, d)
    K/V: (b, s, h, d)

    returns:
    S/P: (b, h, t, s)
    O: (b, t, h, d)
    L/l/m: (b, h, t)
    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, V)

    m = torch.max(S, dim = -1)[0]
    l = torch.sum(torch.exp(S - m[:,:,:, None]), dim = -1)
    L = (m + torch.log(l)) / np.log(2)
    return S, P, O, L


def naive_backward_batched(Q, K, V, O, dO, softmax_scale):
    """
    O = (b, t, h, d)
    dO = (b, t, h, d)
    """
    S = torch.einsum("b t h d, b s h d -> b h t s", Q, K) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    # O = torch.einsum("b h t s, b s h d -> b t h d", P, V)

    D = torch.einsum("b t h d, b t h d -> b h t", O, dO)

    """CHECK THIS!!"""
    dV = torch.einsum("b h t s, b t h d -> b s h d", P, dO)
    dP = torch.einsum("b t h d, b s h d -> b h t s", dO, V)
    
    t = (P * dP).sum(axis = -1)[:,:,:,None]
    dS = P * (dP - t)

    dQ = torch.einsum("b h t s, b s h d -> b t h d", dS, K) * softmax_scale
    dK = torch.einsum("b h t s, b t h d -> b s h d", dS, Q) * softmax_scale
    

    return dQ, dK, dV
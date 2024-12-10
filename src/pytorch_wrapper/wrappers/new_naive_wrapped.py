import torch 
from typing import Optional, Sequence, Tuple, Union
import math
import numpy as np

def force_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# if this causes issues, I think we can just delete this
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op

"""
Reference
csrc/flash_attn/flash_api.cpp to see the official cuda version api
"""



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


def naive_backward_batched(Q, K, V, O, dO, L, softmax_scale):
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


_wrapped_naive_attn_forward = naive_forward_batched_supports
_wrapped_naive_attn_backward = naive_backward_batched


class baseline_AttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qkv): 
        softmax_scale = qkv.shape[-1] ** (-0.5)
        Q,K,V = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()


        B, N, H, D = Q.shape
        assert (D % 8 == 0), "head dimension must be divisible by 8"

        softmax_scale = 1.0 / math.sqrt(D)
        S, P, O, L =  _wrapped_naive_attn_forward(Q,K,V,softmax_scale)

        ctx.save_for_backward(Q, K, V, O, L)
        return O
    
    @staticmethod
    def backward(ctx, dO, *args):
        Q, K, V, O, L = ctx.saved_tensors
        B, N, H, D = Q.shape

        qkv_shape = (B, N, 3, H, D)


        dqkv = torch.empty(qkv_shape, dtype=Q.dtype, device=Q.device)
        assert (D % 8 == 0), "head dimension must be divisible by 8"
            
        dQ,dK,dV = _wrapped_naive_attn_backward(Q, K, V, O, dO, L, 1.0 / math.sqrt(D))
        dqkv[:, :, 0] = dQ
        dqkv[:, :, 1] = dK
        dqkv[:, :, 2] = dV
        

        # you will need to return (,None) x (number of arguments in forward besides qkv that don't need gradients!)
        return dqkv
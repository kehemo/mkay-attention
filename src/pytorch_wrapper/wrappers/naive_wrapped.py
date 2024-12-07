import torch 
from typing import Optional, Sequence, Tuple, Union
import math

def force_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# if this causes issues, I think we can just delete this
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op

"""
Reference
csrc/flash_attn/flash_api.cpp to see the official cuda version api
"""



# @_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _naive_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    q, k, v = [force_contiguous(x) for x in (q, k, v)]

    # IMPLEMENTATION STARTS
    d = q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    S = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale
    P = torch.softmax(S, dim=-1)  # softmax along key dimension
    O = torch.einsum("b h t s, b s h d -> b t h d", P, v)


    # IMPLEMENTATION ENDS
    return S, P, O



# @_torch_custom_op_wrapper("flash_attn::_flash_attn_backward", mutates_args=(), device_types="cuda")
def _naive_attn_backward(
    dO: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    S: torch.Tensor,
    P: torch.Tensor,
    O: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # IMPLEMENTATION STARTS

    dV = torch.einsum("b h t s, b t h d -> b s h d", P, dO)
    dP = torch.einsum("b t h d, b s h d -> b h t s", dO, V)
     
    t = (P * dP).sum(axis = -1)[:,:,:,None]
    dS = P * (dP - t)

    d = Q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d)
    
    dQ = torch.einsum("b h t s, b s h d -> b t h d", dS, K) * softmax_scale
    dK = torch.einsum("b h t s, b t h d -> b s h d", dS, Q) * softmax_scale

    # IMPLEMENTATION ENDS

    return dQ, dK, dV



_wrapped_naive_attn_forward = _naive_attn_forward
_wrapped_naive_attn_backward = _naive_attn_backward


class naive_AttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qkv): 
        softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()

        B, N, H, D = q.shape
        assert (D % 8 == 0), "head dimension must be divisible by 8"

        S, P, O =  _wrapped_naive_attn_forward(q,k,v,)
        ctx.save_for_backward(q, k, v, S, P, O)
        return O
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, S, P, O = ctx.saved_tensors
        B, N, H, D = q.shape
        qkv_shape = (B, N, 3, H, D)


        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        assert (D % 8 == 0), "head dimension must be divisible by 8"
            
        dQ,dK,dV = _wrapped_naive_attn_backward(dout, q, k, v, S, P, O,)
        dqkv[:, :, 0] = dQ
        dqkv[:, :, 1] = dK
        dqkv[:, :, 2] = dV
        

        # you will need to return (,None) x (number of arguments in forward besides qkv that don't need gradients!)
        return dqkv
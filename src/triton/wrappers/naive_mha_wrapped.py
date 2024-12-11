import torch 
import math
import numpy as np


from reference_utils import naive_forward_batched_supports, naive_backward_batched
_wrapped_naive_attn_forward = naive_forward_batched_supports
_wrapped_naive_attn_backward = naive_backward_batched


class naive_AttnQKVPackedFunc(torch.autograd.Function):
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
            
        dQ,dK,dV = _wrapped_naive_attn_backward(Q, K, V, O, dO, 1.0 / math.sqrt(D))
        dqkv[:, :, 0] = dQ
        dqkv[:, :, 1] = dK
        dqkv[:, :, 2] = dV
        

        # you will need to return (,None) x (number of arguments in forward besides qkv that don't need gradients!)
        return dqkv
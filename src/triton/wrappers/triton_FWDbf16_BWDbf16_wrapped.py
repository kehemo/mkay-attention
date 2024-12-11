import torch 
import math
import numpy as np

from reference_utils import naive_forward_batched_supports, naive_backward_batched
_wrapped_naive_attn_forward = naive_forward_batched_supports
_wrapped_naive_attn_backward = naive_backward_batched


from kernels.fwd_FA1_fp32 import flash1_fwd_wrapper as fwd_FA1_fp32_wrapper
from kernels.fwd_FA2_bf16 import flash2_fwd_wrapper as fwd_FA2_bf16_wrapper
from kernels.bwd_FA2_fp32 import flash2_bwd_wrapper as bwd_FA2_fp32_wrapper
from kernels.bwd_FA2_bf16 import flash2_bwd_wrapper as bwd_FA2_bf16_wrapper

class triton_both_AttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qkv): 
        qkv = qkv.to(torch.bfloat16)
        softmax_scale = qkv.shape[-1] ** (-0.5)
        Q,K,V = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        B, N, H, D = Q.shape
        assert (D % 8 == 0), "head dimension must be divisible by 8"

        softmax_scale = 1.0 / math.sqrt(D)
        # O = fwd_FA1_fp32_wrapper(Q, K, V, softmax_scale)
        O, L = fwd_FA2_bf16_wrapper(Q, K, V, softmax_scale)

        

        O = O.to(torch.float32)

        # L = L.to(torch.float32)
        # Q = Q.to(torch.float32)
        # K = K.to(torch.float32)
        # V = V.to(torch.float32)
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, dO, *args):
        Q, K, V, O, L = ctx.saved_tensors
        B, N, H, D = Q.shape
        O = O.to(torch.bfloat16)
        dO = dO.to(torch.bfloat16)

        
        qkv_shape = (B, N, 3, H, D)


        dqkv = torch.empty(qkv_shape, dtype=Q.dtype, device=Q.device)
        assert (D % 8 == 0), "head dimension must be divisible by 8"
            
        # print(f"Q: {Q.dtype}, K: {K.dtype}, V: {V.dtype}, O: {O.dtype}, dO: {dO.dtype}, L: {L.dtype}")
        # dQ,dK,dV = _wrapped_naive_attn_backward(Q, K, V, O, dO, 1.0 / math.sqrt(D))
        D, dQ,dK,dV = bwd_FA2_bf16_wrapper(Q, K, V, O, dO, L, 1.0 / math.sqrt(D))

        dqkv[:, :, 0] = dQ
        dqkv[:, :, 1] = dK
        dqkv[:, :, 2] = dV
        
        print(f"dQ = {dQ[0,:, 0, :]}, dK = {dK[0,:, 0, :]}, dV = {dV[0,:, 0, :]}")
        # assert False, f"dQ = {dQ[0,:, 0, :]}, dK = {dK[0,:, 0, :]}, dV = {dV[0,:, 0, :]}"
        dqkv = dqkv.to(torch.float32)
        # you will need to return (,None) x (number of arguments in forward besides qkv that don't need gradients!)
        return dqkv
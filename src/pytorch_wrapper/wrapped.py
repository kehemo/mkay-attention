import torch 
from typing import Optional, Sequence, Tuple, Union

def force_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# if this causes issues, I think we can just delete this
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op

"""
Reference
csrc/flash_attn/flash_api.cpp to see the official cuda version api
"""



@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    q, k, v = [force_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(q,k,v,)
    return out, softmax_lse, S_dmask, rng_state


@_torch_custom_op_wrapper("flash_attn::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [force_contiguous(x) for x in (dout, q, k, v, out)]
    dq,dk,dv,softmax_d = flash_attn_cuda.bwd( dout, q, k, v, out, softmax_lse, dq, dk, dv,)
    return softmax_d




_wrapped_flash_attn_forward = _flash_attn_forward
_wrapped_flash_attn_backward = _flash_attn_backward


class MKAY_AttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,qkv): 
        softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()

        B, N, H, D = q.shape
        assert (D % 8 == 0), "head dimension must be divisible by 8"

        out, softmax_lse =  _wrapped_flash_attn_forward(q,k,v,)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        return out
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse= ctx.saved_tensors
        B, N, H, D = q.shape
        qkv_shape = (B, N, H, 3, D)


        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        assert (D % 8 == 0), "head dimension must be divisible by 8"
            
        _wrapped_flash_attn_backward(dout,q,k,v,out,softmax_lse,dqkv[:, :, 0],dqkv[:, :, 1],dqkv[:, :, 2],)
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None
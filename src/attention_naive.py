import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import cuda_extension


def attention_pytorch(qkv, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    softmax_scale = 1.0 / math.sqrt(d)


    attn_score_shape = (batch_size * nheads, seqlen, seqlen)
    attn_scores = torch.empty(attn_score_shape, dtype = attn_dtype, device = attn_device)


    """
    ========================================
    Implementation fork
    - they have a long messy way of computing things
    - i think einsum is a little easier to read here, but its up to whatever you like

    NOTE: softmax_scale =/= 1.0 introduces small tolerance issues.
    If you set it to 1.0, the results will be exactly the same... weird.
    ========================================
    """
    
    # technically they are the ground truth >:(
    use_approach_1 = True 
    if use_approach_1:
        # approach 1
        attn_scores_re = attn_scores.clone().detach()
        
        q_re = rearrange(q, "b t h d -> (b h) t d")
        k_re = rearrange(k, "b s h d -> (b h) d s")

        attn_scores_re = torch.baddbmm(attn_scores_re, q_re, k_re, beta = 0, alpha = softmax_scale)
        attn_scores_re = rearrange(attn_scores_re, "(b h) t s -> b h t s", h = nheads)

        attn_scores = attn_scores_re
    else: 
        # approach 2
        attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) * softmax_scale

    attention = torch.softmax(attn_scores, dim = -1) # softmax along key dimension
    output = torch.einsum("b h t s, b s h d -> b t h d", attention, v)

    return output.to(dtype=qkv.dtype)

def attention_torch_checkpoint(qkv):
    attn_dtype, attn_device = qkv.dtype, qkv.device

    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    softmax_scale = 1.0 / math.sqrt(d)

    attn_scores = torch.einsum("b t h d, b s h d -> b h t s", q, k) # * softmax_scale
    output = attn_scores # for now

    return output.to(dtype=qkv.dtype)

def attention_cuda(qkv):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    # q, k, v are of shape (batch_size, seqlen, nheads, head_dim)
    return cuda_extension.attention_forward(q, k, v)


if __name__ == "__main__":

    batch_size = 1
    seqlen = 32

    headdim = 4 # 64
    nheads = 1 # 32

    dtype = torch.bfloat16
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device="cuda")


    torch_output = attention_torch_checkpoint(qkv)
    cuda_output = attention_cuda(qkv)

    print(torch_output)

    print(cuda_output)
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
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    return cuda_extension.attention_forward(q, k, v)
    q = rearrange(q, "b t h d -> (b h) t d")
    k = rearrange(k, "b s h d -> (b h) d s")
    softmax_scale = 1.0 / math.sqrt(d)

    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(
        batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
    )
    scores = rearrange(
        torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
        "(b h) t s -> b h t s",
        h=nheads,
    )

    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=qkv.dtype)


if __name__ == "__main__":

    batch_size = 1
    seqlen = 4

    headdim = 128
    dim = 2048
    nheads = dim // headdim

    dtype = torch.bfloat16
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype)
    output = attention_pytorch(qkv)

    print(f"output: {output}")

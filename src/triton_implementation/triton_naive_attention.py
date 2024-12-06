import torch
import triton
import triton.language as tl

@triton.jit
def attention_triton(
    q, k, v, output, seqlen, d,
    batch_stride, seqlen_stride, nheads_stride, nheads, BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # pointers to the appropriate batch and head
    # (batch_idx, 0, head_idx, 0)
    # these r literally all the same but for the sake of clarity
    q_ptr = q + batch_idx * batch_stride + head_idx * nheads_stride
    k_ptr = k + batch_idx * batch_stride + head_idx * nheads_stride
    v_ptr = v + batch_idx * batch_stride + head_idx * nheads_stride
    out_ptr = output + batch_idx * batch_stride + head_idx * nheads_stride
    iters_needed = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE

    outer_mask = tl.arange(0, BLOCK_SIZE) < seqlen
    total_outer_mask = outer_mask[:, None] & outer_mask[None, :]

    for iteration in range(iters_needed):
        # need to grab the sequences for this particular head, and this iteration of the sequence
        # there reason for tl.arrange(0, BLOCK_SIZE) instead of tl.arrange(0, seqlen) is because you can't use seqlen as a constexpr
        offs_n = iteration * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        # same here, we just mask it out later
        offs_d = tl.arange(0, BLOCK_SIZE)

        # shape of q, k, v is (batch_size, seqlen, nheads, d)
        # the block needs to be of shape (seqlen, d) for a single head as described by head_idx
        q_ptrs = q_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]
        k_ptrs = k_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]
        v_ptrs = v_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]

        output_ptrs = out_ptr + offs_n[:, None] * seqlen_stride + offs_d[None, :]
        mask1 = offs_n < iteration * BLOCK_SIZE + seqlen
        mask2 = offs_d < d
        total_mask = mask1[:, None] & mask2[None, :]
    
        q_block = tl.load(
            q_ptrs, mask = total_mask, other = 0.0
        )
        k_block = tl.load(
            k_ptrs, mask = total_mask, other = 0.0
        )
        v_block = tl.load(
            v_ptrs, mask = total_mask, other = 0.0
        )

        attn_scores = tl.dot(q_block, tl.trans(k_block)) / tl.sqrt(d.to(tl.float32))
        # necessary when BATCH_SIZE is bigger than seqlen
        attn_scores = tl.where(total_outer_mask, attn_scores, float("-inf"))

        # we have to do our own softmax instead of using torch.softmax because we want softmax along key dimension
        max_scores = tl.max(attn_scores, axis = 1, keep_dims=True)
        attn_scores = attn_scores - max_scores
        exp_scores = tl.exp(attn_scores)
        sum_exp_scores = tl.sum(exp_scores, axis=1, keep_dims = True)
        attn_weights = exp_scores / sum_exp_scores


        output_block = tl.dot(attn_weights.to(tl.bfloat16), v_block)

        # Store result
        tl.store(
            output_ptrs,
            output_block,
            mask=total_mask
        )

def attention_triton_launch(qkv):
    batch_size, seqlen, _, nheads, d = qkv.shape

    q, k, v = qkv.unbind(dim=2)

    # Define grid dimensions based on batch size and number of heads
    grid = (batch_size, nheads)


    q_cont = q.contiguous()
    k_cont = k.contiguous()
    v_cont = v.contiguous()

    output = torch.empty_like(q_cont, dtype=torch.bfloat16)
    batch_stride = seqlen * nheads * d
    seqlen_stride = nheads * d
    nheads_stride = d

    attention_triton[grid](
        q_cont, k_cont, v_cont, output,
        seqlen, d, batch_stride, seqlen_stride, nheads_stride, nheads, BLOCK_SIZE=64
    )
    
    output = output.view(batch_size, seqlen, nheads, d)
    return output
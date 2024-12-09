import torch
import triton
import triton.language as tl
from triton.runtime import driver
import time
import math

"""
CUDA_VISIBLE_DEVICES=1 TRITON_PRINT_AUTOTUNING=1 python3 flash.py
"""


def get_cuda_autotune_config():
    num_stages_list = [4, 8]
    num_warps_list = [4, 8, 16]
    Br_list = [64, 128]
    Bc_list = [64, 128]
    configs = []
    for num_stages in num_stages_list:
        for num_warps in num_warps_list:
            for Br in Br_list:
                for Bc in Bc_list:
                    configs.append(triton.Config({'Br': Br, 'Bc': Bc}, num_stages=num_stages, num_warps=num_warps))
    return configs
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['seqlen', 'hdim'],
)
@triton.jit
def attention_triton(
    q, k, v, 
    output, softmax_score,
    qkv_size, seqlen, nheads, hdim: tl.constexpr,
    L, 
    L_batch_stride, L_heads_stride,
    batch_stride, seqlen_stride, nheads_stride, 
    Br : tl.constexpr, Bc : tl.constexpr):
    # compiler hints, i am unsure how to use these
    # tl.max_constancy(q, qkv_size)
    # tl.max_constancy(k, qkv_size)
    # tl.max_constancy(v, qkv_size)
    # tl.max_contiguous(q, qkv_size)
    # tl.max_contiguous(k, qkv_size)
    # tl.max_contiguous(v, qkv_size)

    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    Tr_i = tl.program_id(2)

    # point to (batch_id, 0, head_id, 0)
    q_ptr = q + batch_id * batch_stride + head_id * nheads_stride
    k_ptr = k + batch_id * batch_stride + head_id * nheads_stride
    v_ptr = v + batch_id * batch_stride + head_id * nheads_stride
    out_ptr = output + batch_id * batch_stride + head_id * nheads_stride
    
    # using make_block_ptr as opposed to just constructing the n-dimensional tensor of pointers because it's easier
    # and also idk how I would mask it out otherwise. i think block ptrs take care of that for you
    k_block_ptr = tl.make_block_ptr(
        base = k_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1), # we can assume stride for d is 1 because we called contiguous() before
        offsets = (0, 0), 
        block_shape = (Bc, hdim),
        order = (1, 0) # this doesn't really matter for correctness unless we're on hopper (jesus christ this is not documented)
        # https://www.mengyibai.com/p/order-in-triton-make-block-ptr/
    )

    v_block_ptr = tl.make_block_ptr(
        base = v_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (0, 0),
        block_shape = (Bc, hdim),
        order = (1, 0)
    )

    q_block_ptr = tl.make_block_ptr(
        base = q_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (Tr_i * Br, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    out_block_ptr = tl.make_block_ptr(
        base = out_ptr,
        shape = (seqlen, hdim),
        strides = (seqlen_stride, 1),
        offsets = (Tr_i * Br, 0),
        block_shape = (Br, hdim),
        order = (1, 0)
    )

    # set up L
    L_ptr = L + batch_id * L_batch_stride + head_id * L_heads_stride
    L_offsets = Tr_i * Br + tl.arange(0, Br)
    L_mask = L_offsets < seqlen
    L_ptrs = L_ptr + L_offsets
    
    # set up li, mi
    li = tl.zeros((Br,), dtype=tl.float32)
    mi = tl.full((Br,), -1e9, dtype=tl.float32)

    Qi = tl.load(q_block_ptr)
    Oi = tl.zeros((Br, hdim), dtype=tl.float32)
    Tc = tl.cdiv(seqlen, Bc)

    for j in range(Tc):
        Kj = tl.load(k_block_ptr)
        Vj = tl.load(v_block_ptr)
        
        # Sij: (Br, Bc)
        Sij = (tl.dot(Qi, tl.trans(Kj)) * softmax_score)
        mi_new = tl.maximum(mi, tl.max(Sij, axis = 1))
        P_t_ij = tl.exp(Sij - mi_new[:, None])
        P_t_ij = P_t_ij.to(tl.bfloat16)
        li_new = tl.exp(mi - mi_new) * li + tl.sum(P_t_ij, axis = 1)
        Oi = tl.exp(mi - mi_new)[:, None] * Oi
        Oi = tl.dot(P_t_ij, Vj, Oi) 

        # advance the block pointers
        k_block_ptr = tl.advance(k_block_ptr, (Bc, 0))
        v_block_ptr = tl.advance(v_block_ptr, (Bc, 0))
        mi = mi_new
        li = li_new
        
    Oi = (1 / li[:, None]) * Oi
    Oi = Oi.to(tl.bfloat16)
    Li = mi + tl.log(li)
    Li = Li.to(tl.bfloat16)
    tl.store(out_block_ptr, Oi)
    tl.store(L_ptrs, Li, mask = L_mask)

def attention_triton_launch(qkv):
    q, k, v = qkv.unbind(dim=2)
    q_cont, k_cont, v_cont = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.zeros_like(q_cont, dtype=qkv.dtype)

    B, N, H, D = q_cont.shape
    qkv_size = q.numel()
    L = torch.zeros((B, H, N), dtype=qkv.dtype, device=qkv.device)

    L_batch_stride, L_heads_stride, _ = L.stride()
    batch_stride, seqlen_stride, nheads_stride, _ = q_cont.stride()


    def grid(META):
        Tr = triton.cdiv(N, META['Br'])
        return (B, H, Tr)

    # compile the kernel
    attention_triton[grid](
        q_cont, k_cont, v_cont, 
        output, 1 / math.sqrt(D),
        qkv_size, N, H, D,
        L,
        L_batch_stride, L_heads_stride,
        batch_stride, seqlen_stride, nheads_stride,
    )
    torch.cuda.synchronize()
    output = torch.zeros_like(q_cont, dtype=qkv.dtype)
    L = torch.zeros((B, H, N), dtype=qkv.dtype, device=qkv.device)
    FLOPS = 4 * B * N * H * D * N
    start = time.time()
    attention_triton[grid](
        q_cont, k_cont, v_cont, 
        output, 1 / math.sqrt(D),
        qkv_size, N, H, D,
        L,
        L_batch_stride, L_heads_stride,
        batch_stride, seqlen_stride, nheads_stride,
    )
    torch.cuda.synchronize()
    end = time.time()
    print(f"triton kernel time: {end - start}")
    print(f"TFLOPS: {FLOPS / (end - start) / 1e12}")
    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()
    kernels = {}

    print(f"device stats")
    print(f"NUM_SM {NUM_SM}, NUM_REGS {NUM_REGS}, SIZE_SMEM {SIZE_SMEM}, WARP_SIZE {WARP_SIZE}")
    # output: NUM_SM 84, NUM_REGS 65536, SIZE_SMEM 101376, WARP_SIZE 32
    # 32, 2048, 32, 64
    batch_size = 32
    seq_len = 2048
    n_heads = 32
    head_dim = 64
    
    qkv = torch.randn(batch_size, seq_len, 3, n_heads, head_dim, device = 'cuda', dtype=torch.bfloat16)
    attention_triton_launch(qkv)
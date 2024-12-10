import torch
import triton
import math
from triton.runtime import driver
from triton_flash_attention_2_backwards import flash2_bwd_wrapper, naive_forward_batched_supports, naive_backward_batched

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(8, 11)],
        line_arg='provider',
        line_vals=['triton'], # ANUGRAHHHHHH, add torch here if u wanna see torch results :)
        line_names=['Triton'], # and here
        styles=[('red', '-'), ('blue', '-')],
        ylabel='TFLOPS',
        plot_name='flash-attention-backward-performance',
        args={
            'BATCH': 4,
            'N_HEADS': 32,
            'HEAD_DIM': 64,
        }
    )
)
def benchmark_flash_attention_backward(BATCH, N_HEADS, N_CTX, HEAD_DIM, provider, device='cuda'):
    dtype = torch.float16
    qkv = torch.randn((BATCH, N_CTX, 3, N_HEADS, HEAD_DIM), device=device, dtype=dtype)
    q, k, v = qkv.unbind(dim=2)
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    
    # Forward pass to get necessary inputs for backward
    S, P, O, L, l, m = naive_forward_batched_supports(q, k, v, sm_scale)
    dO = torch.randn_like(O)
    
    if provider == 'triton':
        fn = lambda: flash2_bwd_wrapper(q, k, v, O, dO, L, sm_scale)
    elif provider == 'torch':
        fn = lambda: naive_backward_batched(q, k, v, O, dO, L, sm_scale)
    
    # Warmup
    for _ in range(10):
        fn()
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # Calculate FLOPS (approximation, as backward pass is more complex)
    flops_per_matmul = 2.0 * BATCH * N_HEADS * N_CTX * N_CTX * HEAD_DIM
    total_flops = 5 * flops_per_matmul  # Rough estimate for backward pass
    
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()
    
    print(f"Device stats:")
    print(f"NUM_SM {NUM_SM}, NUM_REGS {NUM_REGS}, SIZE_SMEM {SIZE_SMEM}, WARP_SIZE {WARP_SIZE}")
    
    benchmark_flash_attention_backward.run(save_path=".", print_data=True)
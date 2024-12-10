import torch
import triton
import os
# set environment variable TRITON_PRINT_AUTOTUNING=1
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
from triton_flash_attention_2_forwards import attention_triton_launch

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N_CTX'],
        x_vals=[2**i for i in range(10, 13)],
        line_arg='provider',
        line_vals=['triton'],
        line_names=['Triton'],
        styles=[('red', '-')],
        ylabel='TFLOPS',
        plot_name='fused-attention-performance',
        args={
            'BATCH': 4,
            'N_HEADS': 32,
            'HEAD_DIM': 64,
        }
    )
)
def benchmark_attention(BATCH, N_HEADS, N_CTX, HEAD_DIM, provider, device='cuda'):
    dtype = torch.bfloat16
    qkv = torch.randn((BATCH, N_CTX, 3, N_HEADS, HEAD_DIM), device=device, dtype=dtype)
    
    if provider == 'triton':
        fn = lambda: attention_triton_launch(qkv)
    
    # Warmup
    for _ in range(10):
        fn()
    
    # Benchmark
    ms = triton.testing.do_bench(fn)
    
    # Calculate FLOPS
    flops_per_matmul = 2.0 * BATCH * N_HEADS * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    
    return total_flops * 1e-12 / (ms * 1e-3)

if __name__ == '__main__':
    benchmark_attention.run(save_path='.', print_data=True)
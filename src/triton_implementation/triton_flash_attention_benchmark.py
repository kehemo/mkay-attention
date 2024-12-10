import triton_flash_attention_2_forwards_bench
import triton_flash_attention_2_backwards_bench
import triton_flash_attention_tutorial
import pandas as pd

def main():
    print("Running Triton Flash Attention Benchmark (Forward Pass)")
    fa2_df_forward = triton_flash_attention_2_forwards_bench.benchmark_attention.run(save_path='.', return_df=True)
    print("Running Triton Flash Attention Tutorial Benchmark (Forward Pass)")
    fa2_tutorial_df_forward = triton_flash_attention_tutorial.bench_flash_attention_fwd.run(save_path='.', return_df=True)

    merged_df = pd.merge(fa2_df_forward, fa2_tutorial_df_forward[0], on='N_CTX')
    merged_df['Percent_Speed_FP16'] = (merged_df['Triton'] / merged_df['Triton [FP16]']) * 100
    merged_df = merged_df.drop(columns=['Triton [FP8]'])
    merged_df = merged_df.rename(columns={'Triton [FP16]': 'Triton Tutorial', 'Triton': 'Our Triton'})
    print(merged_df)

    print("Running Triton Flash Attention Benchmark (Backward Pass)")
    fa2_df_backward = triton_flash_attention_2_backwards_bench.benchmark_flash_attention_backward.run(save_path='.', return_df=True)
    print("Running Triton Flash Attention Tutorial Benchmark (Backward Pass)")
    fa2_tutorial_df_backward = triton_flash_attention_tutorial.bench_flash_attention_bwd.run(save_path='.', return_df=True)

    merged_df = pd.merge(fa2_df_backward, fa2_tutorial_df_backward[0], on='N_CTX')
    merged_df['Percent_Speed_FP16'] = (merged_df['Triton'] / merged_df['Triton [FP16]']) * 100
    merged_df = merged_df.drop(columns=['Triton [FP8]'])
    merged_df = merged_df.rename(columns={'Triton [FP16]': 'Triton Tutorial', 'Triton': 'Our Triton'})
    print(merged_df)


if __name__ == "__main__":
    main()
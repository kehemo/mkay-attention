import triton_flash_attention_2_forwards_bench
import triton_flash_attention_2_backwards_bench
import triton_flash_attention_tutorial
import pandas as pd

def main():
    num_runs = 50  # Number of times to run the benchmarks
    forward_results = []
    backward_results = []

    for _ in range(num_runs):
        print("Running Triton Flash Attention Benchmark (Forward Pass)")
        fa2_df_forward = triton_flash_attention_2_forwards_bench.benchmark_attention.run(save_path='.', return_df=True)
        print("Running Triton Flash Attention Tutorial Benchmark (Forward Pass)")
        fa2_tutorial_df_forward = triton_flash_attention_tutorial.bench_flash_attention_fwd.run(save_path='.', return_df=True)

        merged_forward_df = pd.merge(fa2_df_forward, fa2_tutorial_df_forward[0], on='N_CTX')
        merged_forward_df['Percent_Speed_FP16'] = (merged_forward_df['Triton'] / merged_forward_df['Triton [FP16]']) * 100
        merged_forward_df = merged_forward_df.drop(columns=['Triton [FP8]'])
        merged_forward_df = merged_forward_df.rename(columns={'Triton [FP16]': 'Triton Tutorial', 'Triton': 'Our Triton'})
        
        forward_results.append(merged_forward_df)

        print("Running Triton Flash Attention Benchmark (Backward Pass)")
        fa2_df_backward = triton_flash_attention_2_backwards_bench.benchmark_flash_attention_backward.run(save_path='.', return_df=True)
        print("Running Triton Flash Attention Tutorial Benchmark (Backward Pass)")
        fa2_tutorial_df_backward = triton_flash_attention_tutorial.bench_flash_attention_bwd.run(save_path='.', return_df=True)

        merged_backward_df = pd.merge(fa2_df_backward, fa2_tutorial_df_backward[0], on='N_CTX')
        merged_backward_df['Percent_Speed_FP16'] = (merged_backward_df['Triton'] / merged_backward_df['Triton [FP16]']) * 100
        merged_backward_df = merged_backward_df.drop(columns=['Triton [FP8]'])
        merged_backward_df = merged_backward_df.rename(columns={'Triton [FP16]': 'Triton Tutorial', 'Triton': 'Our Triton'})

        backward_results.append(merged_backward_df)

    # Calculate averages for forward results
    average_forward_result = pd.concat(forward_results).groupby('N_CTX').mean().reset_index()
    print("Average Forward Results:")
    print(average_forward_result)

    # Calculate averages for backward results
    average_backward_result = pd.concat(backward_results).groupby('N_CTX').mean().reset_index()
    print("Average Backward Results:")
    print(average_backward_result)

if __name__ == "__main__":
    main()

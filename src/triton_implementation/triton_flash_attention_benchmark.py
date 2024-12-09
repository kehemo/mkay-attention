import triton_flash_attention_2
import triton_flash_attention_tutorial
import pandas as pd

def main():
    fa2_df = triton_flash_attention_2.benchmark_attention.run(save_path='.', return_df=True)
    fa2_tutorial_df = triton_flash_attention_tutorial.bench_flash_attention.run(save_path='.', return_df=True)
    # Merge dataframes on N_CTX
    merged_df = pd.merge(fa2_df, fa2_tutorial_df[0], on='N_CTX')
    
    # Calculate the percent speed of flash attention 2 relative to the tutorial for each sequence length
    merged_df['Percent_Speed_FP16'] = (merged_df['Triton'] / merged_df['Triton [FP16]']) * 100

    # remove the fp8 column
    merged_df = merged_df.drop(columns=['Triton [FP8]'])

    # rename Triton [FP16] to Triton Tutorial
    merged_df = merged_df.rename(columns={'Triton [FP16]': 'Triton Tutorial', 'Triton': 'Our Triton'})

    print(merged_df)

if __name__ == "__main__":
    main()
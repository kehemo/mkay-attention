#include <torch/extension.h>
#include <vector>
#include "kernel.hu"

std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int batch_size = q.size(0);
    int seqlen = q.size(1);
    int nheads = q.size(2);
    int head_dim = q.size(3);

    int64_t batch_stride = q.stride(0);
    int64_t seq_stride = q.stride(1);
    int64_t head_stride = q.stride(2);
    int64_t dim_stride = q.stride(3);

    // malloc output tensor
    num *out;
    uint32_t num_output_elements = batch_size * nheads * seqlen * seqlen;
    size_t output_size = num_output_elements * sizeof(num);
    CUDA_CHECK(cudaMalloc(&out, output_size));
    // printf("malloc'd output_size = %d bytes\n", num_output_elements);

    // launch kernel
    const uint32_t num_calls = batch_size * nheads;
    const uint32_t n_tiles_i = (seqlen + tile_dim - 1) / tile_dim;
    const uint32_t n_tiles_j = (seqlen + tile_dim - 1) / tile_dim;
    dim3 num_blocks_k1 = dim3(num_calls, n_tiles_i, n_tiles_j);
    dim3 thread_grid = dim3(num_threads_axis, num_threads_axis);

    // printf("launching kernel with num_blocks = (%d, %d, %d)\n", num_blocks.x, num_blocks.y, num_blocks.z);
    compute_attn_scores<<<num_blocks, thread_grid>>>(
        q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(),
        out,
        batch_size, nheads, seqlen, head_dim,
        batch_stride, seq_stride, head_stride, dim_stride);
    cudaDeviceSynchronize();

    // create output tensor
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    auto out_tensor = torch::from_blob(P, {batch_size, nheads, seqlen, seqlen}, options);
    return {out_tensor};
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

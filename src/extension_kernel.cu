#include <torch/extension.h>
#include <vector>
#include "kernel.hu"

std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int batch_size = q.size(0);
    int seqlen = q.size(1);
    int nheads = q.size(2);
    int head_dim = q.size(3);

    int64_t batch_stride_qkv = q.stride(0);
    int64_t seq_stride_qkv = q.stride(1);
    int64_t head_stride_qkv = q.stride(2);
    int64_t dim_stride_qkv = q.stride(3);

    // malloc output tensors
    num *O;
    size_t output_size = sizeof(num) * batch_size * nheads * seqlen * seqlen;
    CUDA_CHECK(cudaMalloc(&O, output_size));

    num *S;
    size_t attn_scores_size = sizeof(num) * batch_size * nheads * seqlen * seqlen;
    CUDA_CHECK(cudaMalloc(&S, attn_scores_size));

    num *P;
    size_t softmax_scores_size = sizeof(num) * batch_size * nheads * seqlen * seqlen;
    CUDA_CHECK(cudaMalloc(&P, softmax_scores_size));

    const uint32_t num_batches_heads = batch_size * nheads;

    // launch kernel 1
    const uint32_t n_tiles_i = (seqlen + tile_dim - 1) / tile_dim;
    const uint32_t n_tiles_j = (seqlen + tile_dim - 1) / tile_dim;
    dim3 num_blocks_k1 = dim3(num_batches_heads, n_tiles_i, n_tiles_j);
    dim3 thread_dim_k1 = dim3(num_threads_axis, num_threads_axis);

    // printf("launching kernel with num_blocks = (%d, %d, %d)\n", num_blocks.x, num_blocks.y, num_blocks.z);
    compute_attn_scores<<<num_blocks_k1, thread_dim_k1>>>(
        q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(),
        S,
        batch_size, nheads, seqlen, head_dim,
        batch_stride_qkv, seq_stride_qkv, head_stride_qkv, dim_stride_qkv);
    cudaDeviceSynchronize();

    // launch kernel 2
    dim3 num_blocks_k2 = dim3(batch_size, nheads);
    int num_threads_k2 = seqlen > 1024 ? 1024 : seqlen;
    dim3 thread_grid_k2 = dim3(num_threads_k2);

    compute_attn_softmax<<<num_blocks_k2, thread_grid_k2>>>(
        S,
        P,
        batch_size, nheads, seqlen, head_dim);

    // launch kernel 3
    const uint32_t n_tiles_seqlen = (seqlen + seqlen_tile_k3 - 1) / seqlen_tile_k3;
    const uint32_t n_tiles_head_dim = (head_dim + head_dim_tile_k3 - 1) / head_dim_tile_k3;

    dim3 num_blocks_k3 = dim3(num_batches_heads, n_tiles_seqlen, n_tiles_head_dim);
    dim3 thread_dim_k3 = dim3(num_threads_axis, num_threads_axis);

    compute_attn_output<<<num_blocks_k3, thread_dim_k3>>>(
        P, v.data_ptr<num>(),
        O,
        batch_size, nheads, seqlen, head_dim,
        batch_stride_qkv, seq_stride_qkv, head_stride_qkv, dim_stride_qkv);

    // free memory
    CUDA_CHECK(cudaFree(S));
    CUDA_CHECK(cudaFree(P));

    // create output tensor
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    // auto out_tensor = torch::from_blob(P, {batch_size, nheads, seqlen, seqlen}, options);
    auto out_tensor = torch::from_blob(O, {batch_size, seqlen, nheads, head_dim}, options);
    return {out_tensor};
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

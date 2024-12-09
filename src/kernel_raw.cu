#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda_bf16.h>
#include "kernel.hu"

struct launch_raw_result
{
    num *data;
    size_t len;
};

launch_raw_result launch_attention_raw(num *q, num *k, num *v,
                                       int batch_size, int seqlen, int nheads, int head_dim)
{

    int64_t batch_stride_qkv = seqlen * nheads * head_dim;
    int64_t seq_stride_qkv = nheads * head_dim;
    int64_t head_stride_qkv = head_dim;
    int64_t dim_stride_qkv = 1;

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
        q, k, v,
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
        P, v,
        O,
        batch_size, nheads, seqlen, head_dim,
        batch_stride_qkv, seq_stride_qkv, head_stride_qkv, dim_stride_qkv);

    // free memory
    CUDA_CHECK(cudaFree(S));
    CUDA_CHECK(cudaFree(P));

    return {O, output_size};
}
/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////
num *read_input(std::string name)
{

    std::ifstream file(name, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << name << " " << size << std::endl;
    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size))
    {
        num *cuda_buf;
        CUDA_CHECK(cudaMalloc(&cuda_buf, buffer.size()));
        CUDA_CHECK(cudaMemcpy(cuda_buf, buffer.data(), buffer.size(), cudaMemcpyHostToDevice));
        file.close();
        return cuda_buf;
    }
    else
    {
        throw "could not read";
    }
}

struct test_config
{
    int batch_size;
    int seqlen;
    int nheads;
    int headdim;
};

int main(void)
{
    std::vector<test_config> configs = {{32, 128, 32, 64}};
    for (auto config : configs)
    {
        std::string test_pref = "test_" + std::to_string(config.batch_size) + "x" + std::to_string(config.seqlen) + "x" + std::to_string(config.nheads) + "x" + std::to_string(config.headdim);
        auto q = read_input(test_pref + "_q.bin");
        auto k = read_input(test_pref + "_k.bin");
        auto v = read_input(test_pref + "_v.bin");
        auto result = launch_attention_raw(q, k, v, config.batch_size, config.seqlen, config.nheads, config.headdim);
        std::cout << result.len << std::endl;
        std::vector<char> out_buffer(result.len);
        std::cout << result.data << std::endl;
        CUDA_CHECK(cudaMemcpy(out_buffer.data(), result.data, result.len, cudaMemcpyDeviceToHost));
        std::ofstream out_file("out/" + test_pref + "_o.bin");
        out_file.write(out_buffer.data(), out_buffer.size());
        out_file.flush();
        out_file.close();
    }
}
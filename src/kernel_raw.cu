#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
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
    size_t output_size = sizeof(num) * batch_size * nheads * seqlen * head_dim;
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

launch_raw_result bench_flash_attention_raw(num *q, num *k, num *v,
                                            int batch_size, int seqlen, int nheads, int head_dim, int target_time_ms)
{
    attention_params params = {0};
    params.batch_stride = seqlen * nheads * head_dim;
    params.token_stride = nheads * head_dim;
    params.head_stride = head_dim;
    params.dim_stride = 1;
    params.batch_size = batch_size;
    params.seqlen = seqlen;
    params.nheads = nheads;
    params.head_dim = head_dim;
    params.softmax_scale = rsqrtf(head_dim);

    // malloc output tensors
    num *l;
    size_t lm_size = sizeof(num) * params.batch_size * params.nheads * params.seqlen;
    CUDA_CHECK(cudaMalloc(&l, lm_size));

    num *m;
    CUDA_CHECK(cudaMalloc(&m, lm_size));

    num *o;
    size_t output_size = sizeof(num) * batch_size * nheads * seqlen * head_dim;
    CUDA_CHECK(cudaMalloc(&o, output_size));

    double ops = 4.0 * batch_size * seqlen * seqlen * nheads * head_dim;
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        launch_flash_attention(q, k, v, o, l, m, params);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    std::cout << "Time: " << best_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (ops / best_time_ms / 1e9) << " TFLOP/s" << std::endl;

    CUDA_CHECK(cudaFree(l));
    CUDA_CHECK(cudaFree(m));
    return {o, output_size};
}

launch_raw_result launch_flash_attention_raw(num *q, num *k, num *v,
                                             int batch_size, int seqlen, int nheads, int head_dim)
{
    attention_params params = {0};
    params.batch_stride = seqlen * nheads * head_dim;
    params.token_stride = nheads * head_dim;
    params.head_stride = head_dim;
    params.dim_stride = 1;
    params.batch_size = batch_size;
    params.seqlen = seqlen;
    params.nheads = nheads;
    params.head_dim = head_dim;
    params.softmax_scale = rsqrtf(head_dim);

    // malloc output tensors
    num *l;
    size_t lm_size = sizeof(num) * params.batch_size * params.nheads * params.seqlen;
    CUDA_CHECK(cudaMalloc(&l, lm_size));

    num *m;
    CUDA_CHECK(cudaMalloc(&m, lm_size));

    num *o;
    size_t output_size = sizeof(num) * batch_size * nheads * seqlen * head_dim;
    CUDA_CHECK(cudaMalloc(&o, output_size));
    launch_flash_attention(q, k, v, o, l, m, params);

    CUDA_CHECK(cudaFree(l));
    CUDA_CHECK(cudaFree(m));
    return {o, output_size};
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
    std::ifstream test_sizes("test_sizes.csv");
    std::string config_string;
    while (getline(test_sizes, config_string))
    {
        std::istringstream ss(config_string);
        std::vector<int> params;
        std::string param;
        while (getline(ss, param, ','))
        {
            params.push_back(std::stoi(param));
        }
        test_config config = {params[0], params[1], params[2], params[3]};
        std::string test_pref = "test_" + std::to_string(config.batch_size) + "x" + std::to_string(config.seqlen) + "x" + std::to_string(config.nheads) + "x" + std::to_string(config.headdim);
        auto q = read_input(test_pref + "_q.bin");
        auto k = read_input(test_pref + "_k.bin");
        auto v = read_input(test_pref + "_v.bin");
        auto result = bench_flash_attention_raw(q, k, v, config.batch_size, config.seqlen, config.nheads, config.headdim, 1000);
        std::vector<char> out_buffer(result.len);
        CUDA_CHECK(cudaMemcpy(out_buffer.data(), result.data, result.len, cudaMemcpyDeviceToHost));
        std::ofstream out_file("out/" + test_pref + "_o.bin");
        out_file.write(out_buffer.data(), out_buffer.size());
        out_file.flush();
        out_file.close();
    }
}
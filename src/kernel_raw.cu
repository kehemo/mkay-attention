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

    int64_t batch_stride = seqlen * nheads * head_dim;
    int64_t seq_stride = nheads * head_dim;
    int64_t head_stride = head_dim;
    int64_t dim_stride = 1;

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
    dim3 num_blocks = dim3(num_calls, n_tiles_i, n_tiles_j);
    dim3 thread_grid = dim3(num_threads_axis, num_threads_axis);

    // printf("launching kernel with num_blocks = (%d, %d, %d)\n", num_blocks.x, num_blocks.y, num_blocks.z);
    compute_attn_scores<<<num_blocks, thread_grid>>>(
        q, k, v,
        out,
        batch_size, nheads, seqlen, head_dim,
        batch_stride, seq_stride, head_stride, dim_stride);
    cudaDeviceSynchronize();
    return {out, output_size};
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
    std::vector<test_config> configs = {{2, 32, 64, 32}};
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
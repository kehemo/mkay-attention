#include <torch/extension.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

// Method 1: Using ofstream directly
bool writeToFile(const char* filename, const void* data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    file.write(static_cast<const char*>(data), size);
    return file.good();
}
void cuda_check(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x)                        \
    do                                       \
    {                                        \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)
typedef at::BFloat16 num;

__global__ void add_one_kernel(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0;
    }
}


////////////////////////////////////////////////////////////////////////////////
///  YOU DO NOT NEED TO MODIFY THE CODE ABOVE HERE (unless you want to).     ///
////////////////////////////////////////////////////////////////////////////////



/// <--- your code here --->

constexpr int tile_dim = 32;
constexpr int num_threads_axis = 32;

__global__ void compute_attn_scores(
    num *q, const num *k, const num *v,  
    num *scores, 
    int batch_size, int nheads, int seqlen, int head_dim,
    int64_t batch_stride, int64_t token_stride, int64_t head_stride, int64_t dim_stride)
{
    /*
    INPUTS:
    q,k,v === (batch_size, seqlen, nheads, head_dim) 
    OUTPUTS:
    scores === (batch_size, nheads, seqlen, seqlen)    
    NOTES:
    Computes q @ k.T along the seqlen dimension
    */

    uint32_t block_id = blockIdx.x;
    uint32_t batch_id = block_id / nheads;
    uint32_t head_id = block_id % nheads;


    uint32_t tile_i = blockIdx.y;
    uint32_t tile_j = blockIdx.z;
    uint32_t thread_i = threadIdx.x;
    uint32_t thread_j = threadIdx.y;

    for (int row = thread_i; row < tile_dim; row += num_threads_axis) {
        for (int col = thread_j; col < tile_dim; col += num_threads_axis) {
            // compute "score[i][j] = sum_k q[i][d] * k[j][d]"
            num sum = 0.0;
            int i = tile_i * tile_dim + row;
            int j = tile_j * tile_dim + col;

            for (int d = 0; d < head_dim; d++) {

                uint32_t q_idx = batch_id * batch_stride + i * token_stride + head_id * head_stride + d * dim_stride;
                uint32_t k_idx = batch_id * batch_stride + j * token_stride + head_id * head_stride + d * dim_stride;

                num q_val = q[q_idx];
                num k_val = k[k_idx];


                sum += q_val * k_val;
            }


            uint32_t batch_stride_out = nheads * seqlen * seqlen;
            uint32_t head_stride_out = seqlen * seqlen;
            uint32_t token_stride_out = seqlen;
            uint32_t o_idx = batch_id * batch_stride_out + head_id * head_stride_out + i * token_stride_out + j;
            scores[o_idx] = sum;

        }
    }
}



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

    size_t num_input_elements = batch_size * seqlen * nheads * head_dim;
    printf("Looks like we have %0d * %0d * %0d * %0d = %0d input elements per tensor\n",
        batch_size,
        seqlen,
        nheads,
        head_dim,
        num_input_elements
    );

    size_t input_size = num_input_elements * sizeof(num);
    if (writeToFile("out/q.bin", &q, input_size)) {
        std::cout << "Write Q successful\n";
    }
    if (writeToFile("out/k.bin", &k, input_size)) {
        std::cout << "Write K successful\n";
    }
    if (writeToFile("out/v.bin", &v, input_size)) {
        std::cout << "Write V successful\n";
    }

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
        q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(), 
        out, 
        batch_size, nheads, seqlen, head_dim,
        batch_stride, seq_stride, head_stride, dim_stride);
    cudaDeviceSynchronize();

    num *out_cpu = (num*) malloc(output_size);
    CUDA_CHECK(cudaMemcpy(
        out_cpu,
        out,
        output_size,
        cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    if (writeToFile("out/out.bin", &out_cpu, num_input_elements)) {
        std::cout << "Write out.bin successful\n";
    }

    // create output tensor
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    auto out_tensor = torch::from_blob(out, {batch_size, nheads, seqlen, seqlen}, options);
    return {out_tensor};
}



/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

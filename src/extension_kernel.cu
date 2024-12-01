#include <torch/extension.h>
#include <vector>

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
    const num *q, const num *k, const num *v,  
    num *scores, 
    int batch_size, int nheads, int seqlen, int head_dim)
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

            for (int d = 0; d < head_dim; d++) { // FIX THIS LATER

                uint32_t batch_stride = seqlen * nheads * head_dim;
                uint32_t token_stride = nheads * head_dim;
                uint32_t head_stride = head_dim;

                uint32_t q_idx = batch_id * batch_stride + i * token_stride + head_id * head_stride + d;
                uint32_t k_idx = batch_id * batch_stride + j * token_stride + head_id * head_stride + d;

                num q_val = q[q_idx];
                num k_val = k[k_idx];


                sum += q_val * k_val;
            }


            uint32_t batch_stride_out = nheads * seqlen * seqlen;
            uint32_t head_stride_out = seqlen * seqlen;
            uint32_t token_stride_out = seqlen;
            uint32_t o_idx = batch_id * batch_stride_out + head_id * head_stride_out + i * token_stride_out + j;
            scores[o_idx] = sum;
            // scores[o_idx] += 1.0;

        }
    }
}



std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int batch_size = q.size(0);
    int seqlen = q.size(1);
    int nheads = q.size(2);
    int head_dim = q.size(3);

    // malloc output tensor
    num *out;
    uint32_t num_output_elements = batch_size * nheads * seqlen * seqlen;
    size_t output_size = num_output_elements * sizeof(num);
    CUDA_CHECK(cudaMalloc(&out, output_size));
    std::cout << "malloc'd output_size = " << output_size << std::endl;

    // launch kernel

    const uint32_t num_calls = batch_size * nheads;
    const uint32_t n_tiles_i = (seqlen + tile_dim - 1) / tile_dim;
    const uint32_t n_tiles_j = (seqlen + tile_dim - 1) / tile_dim;
    dim3 num_blocks = dim3(num_calls, n_tiles_i, n_tiles_j);
    dim3 thread_grid = dim3(num_threads_axis, num_threads_axis);

    printf("launching kernel with num_blocks = (%d, %d, %d)\n", num_blocks.x, num_blocks.y, num_blocks.z);
    compute_attn_scores<<<num_blocks, thread_grid>>>(
        q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(), 
        out, 
        batch_size, nheads, seqlen, head_dim);
    cudaDeviceSynchronize();


    // create output tensor
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    auto out_tensor = torch::from_blob(out, {batch_size, nheads, seqlen, seqlen}, options);
    return {out_tensor};
}



/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

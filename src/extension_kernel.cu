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

/// <--- your code here --->

__global__ void attention_forward(const num *q, const num *k, const num *v, num *o, int batch_size, int nheads, int seqlen, int head_dim)
{
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////
std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int batch_size = q.size(0);
    int seqlen = q.size(1);
    int nheads = q.size(2);
    int head_dim = q.size(3);
    num *o;
    CUDA_CHECK(cudaMalloc(&o, batch_size * seqlen * nheads * head_dim * sizeof(num)));
    int blocks = 1;
    int threads = 1;
    attention_forward<<<blocks, threads>>>(q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(), o, batch_size, nheads, seqlen, head_dim);
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    auto o_tensor = torch::from_blob(o, {batch_size, seqlen, nheads, head_dim}, options);
    return {o_tensor};
}

// Wrapper function.
std::vector<torch::Tensor> my_cuda_function(torch::Tensor input_tensor)
{
    auto data = input_tensor.data_ptr<float>();
    auto size = input_tensor.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();

    return {input_tensor};
}

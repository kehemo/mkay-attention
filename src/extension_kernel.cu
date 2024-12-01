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

__global__ void attention_forward(const num *q, const num *k, const num *v, num *o, int batch_size, int nheads, int seqlen, int head_dim)
{
}


std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
{
    int batch_size = q.size(0);
    int seqlen = q.size(1);
    int nheads = q.size(2);
    int head_dim = q.size(3);

    // malloc output array
    num *o;
    CUDA_CHECK(cudaMalloc(&o, batch_size * seqlen * nheads * head_dim * sizeof(num)));

    // launch your kernels
    attention_forward<<<1, 1>>>(q.data_ptr<num>(), k.data_ptr<num>(), v.data_ptr<num>(), o, batch_size, nheads, seqlen, head_dim);
    // cudaDeviceSynchronize();
    // in case you want multiple kernels

    // Create a tensor from the output array
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16);
    auto o_tensor = torch::from_blob(o, {batch_size, seqlen, nheads, head_dim}, options);
    return {o_tensor};
}



/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

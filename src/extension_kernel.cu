#include <torch/extension.h>
#include <vector>

typedef at::BFloat16 num;

__global__ void add(num *a, num *b, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        a[idx] += b[idx];
    }
}

__global__ void attention_forward(num *q, num *k, num *v, num *out)
{
}

// std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v)
// {
// }

// Wrapper function.
void launch_add(torch::Tensor a, torch::Tensor b)
{
    auto size = a.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    add<<<blocks, threads>>>(a.data_ptr<num>(), b.data_ptr<num>(), size);
    cudaDeviceSynchronize();
}
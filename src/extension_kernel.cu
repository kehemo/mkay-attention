#include <torch/extension.h>
#include <vector>

typedef at::BFloat16 num;

__global__ void add_one(num *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] += 1.0;
    }
}

// Wrapper function.
void launch_add_one(torch::Tensor input_tensor)
{
    auto data = input_tensor.data_ptr<num>();
    auto size = input_tensor.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    add_one<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();
}
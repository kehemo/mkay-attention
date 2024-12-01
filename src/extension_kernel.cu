#include <torch/extension.h>
#include <vector>

__global__ void add_one_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] += 1.0;
    }
}

// Wrapper function.
std::vector<torch::Tensor> my_cuda_function(torch::Tensor input_tensor) {
    auto data = input_tensor.data_ptr<float>();
    auto size = input_tensor.numel();

    int threads = 1024;
    int blocks = (size + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(data, size);
    cudaDeviceSynchronize();

    return {input_tensor};
}

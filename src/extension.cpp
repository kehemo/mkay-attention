#include <torch/extension.h>
#include <vector>

// Declaration of the kernel function (defined in extension_kernel.cu).

std::vector<torch::Tensor> launch_attention_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attention_forward", &launch_attention_forward, "My CUDA Function");
}

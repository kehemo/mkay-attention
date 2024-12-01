#include <torch/extension.h>
#include <vector>

// Declaration of the kernel function (defined in extension_kernel.cu).
void launch_add_one(torch::Tensor input_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("my_cuda_function", &launch_add_one, "My CUDA Function");
}
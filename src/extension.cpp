#include <torch/extension.h>
#include <vector>

// Declaration of the kernel function (defined in extension_kernel.cu).
std::vector<torch::Tensor> my_cuda_function(torch::Tensor input_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_cuda_function", &my_cuda_function, "My CUDA Function");
}

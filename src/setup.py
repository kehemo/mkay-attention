from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='cuda_extension',
            sources=['extension.cpp', 'extension_kernel.cu'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)

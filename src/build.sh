#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Build code.
# nvcc -O3 vector_add.cu -o ${CTR_BUILD_DIR}/vector_add

# echo "Move benchmarks over"
# cp -r benchmarks ${CTR_BUILD_DIR}/benchmarks



# Build it.
export TORCH_CUDA_ARCH_LIST='8.6'
python3 setup.py build

# Copy built module.
cp build/lib.linux-x86_64-3.10/cuda_extension.cpython-310-x86_64-linux-gnu.so ${CTR_BUILD_DIR}/cuda_extension.so

# Copy example.
cp example.py ${CTR_BUILD_DIR}/ 

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

# Copy example.
cp test*.bin ${CTR_BUILD_DIR}/
nvcc -O3 kernel_raw.cu -o ${CTR_BUILD_DIR}/kernel
#!/bin/bash

set -e
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
python3 generate_attention_data.py
cp test*.bin ${CTR_BUILD_DIR}/
cp test_sizes.csv ${CTR_BUILD_DIR}/
cp flash_attention_ref.py ${CTR_BUILD_DIR}/
nvcc -O3 --use_fast_math -gencode arch=compute_86,code=sm_86 kernel_raw.cu -o ${CTR_BUILD_DIR}/kernel
echo "Finished build"
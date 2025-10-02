#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

export EXECUTORCH_ROOT="$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

# Update tokenizers submodule
pushd $EXECUTORCH_ROOT/extension/llm/tokenizers
echo "Update tokenizers submodule"
git submodule update --init
popd

# Install ET with CMake
cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=OFF \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_TORCHAO=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -Bcmake-out .
cmake --build cmake-out -j16 --config Release --target install

# Install llama runner with torchao
cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_BUILD_TYPE=Release \
    -Bcmake-out/examples/models/llama \
    examples/models/llama
cmake --build cmake-out/examples/models/llama -j16 --config Release

# Download stories llama110m artifacts
download_stories_model_artifacts

echo "Creating tokenizer.bin"
$PYTHON_EXECUTABLE -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin

# Export model
LLAMA_CHECKPOINT=stories110M.pt
LLAMA_PARAMS=params.json
MODEL_OUT=model.pte
TOKENIZER=tokenizer.bin

# Set low-bit quantization parameters
QLINEAR_BITWIDTH=3 # Can be 1-8
QLINEAR_GROUP_SIZE=128 # Must be multiple of 16
QEMBEDDING_BITWIDTH=4 # Can be 1-8
QEMBEDDING_GROUP_SIZE=32 # Must be multiple of 16

${PYTHON_EXECUTABLE} -m extension.llm.export.export_llm \
    base.checkpoint="${LLAMA_CHECKPOINT:?}" \
    base.params="${LLAMA_PARAMS:?}" \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    export.output_name="${MODEL_OUT}" \
    quantization.qmode="torchao:8da${QLINEAR_BITWIDTH}w" \
    quantization.group_size=${QLINEAR_GROUP_SIZE} \
    quantization.embedding_quantize=\"torchao:${QEMBEDDING_BITWIDTH},${QEMBEDDING_GROUP_SIZE}\" \
    model.dtype_override=fp32

# Test run
./cmake-out/examples/models/llama/llama_main --model_path=$MODEL_OUT --tokenizer_path=$TOKENIZER --prompt="Once upon a time,"

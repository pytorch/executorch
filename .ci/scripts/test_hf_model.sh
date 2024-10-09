#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# Input parameter: Hugging Face model repo (e.g., 'google/gemma-2b')
HF_MODEL_REPO=$1
UPLOAD_DIR=${2:-}
DTYPE=${3:-"float32"}

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python
fi
which "${PYTHON_EXECUTABLE}"

# Extract the model name from the HF_MODEL_REPO by splitting on '/' and replacing '_' with '-'
ET_MODEL_NAME=$(echo "$HF_MODEL_REPO" | awk -F'/' '{print $2}' | sed 's/_/-/g')
# Add the suffix "_xnnpack_fp32" to the model name (currently supported delegate and dtype)
OUT_ET_MODEL_NAME="${ET_MODEL_NAME}_xnnpack_fp32"

# Files to be handled
TOKENIZER_FILE="tokenizer.model"
OUT_TOKENIZER_BIN_FILE="tokenizer.bin"

# Download the tokenizer model using Hugging Face hub
DOWNLOADED_TOKENIZER_FILE_PATH=$(${PYTHON_EXECUTABLE} -c "
from huggingface_hub import hf_hub_download
# Download the tokenizer file from the Hugging Face Hub
downloaded_path = hf_hub_download(
    repo_id='${HF_MODEL_REPO}',
    filename='${TOKENIZER_FILE}'
)
print(downloaded_path)
")

# Check if the tokenizer file was successfully downloaded
if [ -f "$DOWNLOADED_TOKENIZER_FILE_PATH" ]; then
    echo "${TOKENIZER_FILE} downloaded successfully at: $DOWNLOADED_TOKENIZER_FILE_PATH"

    # Convert the tokenizer to binary using the Python module
    echo "Convert the tokenizer to binary format"
    "${PYTHON_EXECUTABLE}" -m extension.llm.tokenizer.tokenizer -t "$DOWNLOADED_TOKENIZER_FILE_PATH" -o "./${OUT_TOKENIZER_BIN_FILE}"
    ls "./${OUT_TOKENIZER_BIN_FILE}"
else
    echo "Failed to download ${TOKENIZER_FILE} from ${HF_MODEL_REPO}."
    exit 1
fi

# Export the Hugging Face model
echo "Export the Hugging Face model ${HF_MODEL_REPO} to ExecuTorch"
"${PYTHON_EXECUTABLE}" -m extension.export_util.export_hf_model -hfm="$HF_MODEL_REPO" -o "$OUT_ET_MODEL_NAME" -d "$DTYPE"
ls -All "./${OUT_ET_MODEL_NAME}.pte"

if [ -n "$UPLOAD_DIR" ]; then
    echo "Preparing for uploading generated artifacs"
    zip -j model.zip "${OUT_ET_MODEL_NAME}.pte" "${OUT_TOKENIZER_BIN_FILE}"
    mkdir -p "${UPLOAD_DIR}"
    mv model.zip "${UPLOAD_DIR}"
fi

if [ "$(uname)" == "Darwin" ]; then
    CMAKE_JOBS=$(( $(sysctl -n hw.ncpu) - 1 ))
else
    CMAKE_JOBS=$(( $(nproc) - 1 ))
fi

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out .
    cmake --build cmake-out -j "${CMAKE_JOBS}" --target install --config Release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    dir="examples/models/llama2"
    retry cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_XNNPACK=ON \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j "${CMAKE_JOBS}" --config Release
}

# cmake_install_executorch_libraries
# cmake_build_llama_runner

# ./cmake-out/examples/models/llama2/llama_main --model_path="${OUT_ET_MODEL_NAME}.pte" --tokenizer_path="${OUT_TOKENIZER_BIN_FILE}" --prompt="My name is"

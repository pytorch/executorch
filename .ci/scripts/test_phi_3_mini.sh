#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

BUILD_TYPE=${1:-Debug}
BUILD_DIR=${3:-cmake-out}
MODEL_DIR=examples/models/phi-3-mini

echo "Building with BUILD_TYPE: $BUILD_TYPE, BUILD_DIR: $BUILD_DIR"

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
    PYTHON_EXECUTABLE=python3
fi

# Number of processes for a parallel build
NPROC=8
if hash nproc &> /dev/null; then NPROC=$(nproc); fi

cmake_install_executorch_libraries() {
  rm -rf cmake-out
  cmake --preset llm -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
  cmake --build cmake-out -j16 --target install --config ${BUILD_TYPE}
}

cmake_build_phi_3_mini() {
  cmake -DCMAKE_PREFIX_PATH=${BUILD_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -B${BUILD_DIR}/${MODEL_DIR} \
      ${MODEL_DIR}

  cmake --build ${BUILD_DIR}/${MODEL_DIR} -j${NPROC} --config ${BUILD_TYPE}
}

# Download and convert tokenizer.model
prepare_tokenizer() {
  echo "Downloading and converting tokenizer.model"
  wget -O tokenizer.model "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/tokenizer.model?download=true"
  $PYTHON_EXECUTABLE -m pytorch_tokenizers.tools.llama2c.convert -t tokenizer.model -o tokenizer.bin
}

# Export phi-3-mini model to pte
export_phi_3_mini () {
  echo "Exporting phi-3-mini. This will take a few minutes"
  $PYTHON_EXECUTABLE -m executorch.examples.models.phi-3-mini.export_phi-3-mini -c "4k" -s 128 -o phi-3-mini.pte
}

run_and_verify() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run phi-3-mini runner at ${NOW}"
    if [[ ! -f "phi-3-mini.pte" ]]; then
        echo "Export failed. Abort"
        exit 1
    fi
    if [[ ! -f "tokenizer.bin" ]]; then
        echo "tokenizer.bin is missing."
        exit 1
    fi

    ${BUILD_DIR}/${MODEL_DIR}/phi_3_mini_runner \
    --model_path=phi-3-mini.pte \
    --tokenizer_path=tokenizer.bin \
    --seq_len=60 \
    --temperature=0 \
    --prompt="<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is the capital of France?<|end|>
<|assistant|>" > result.txt

    # verify result.txt
    RESULT=$(cat result.txt)
    EXPECTED_RESULT="The capital of France is Paris."
    if [[ "${RESULT}" == *"${EXPECTED_RESULT}"* ]]; then
        echo "Expected result prefix: ${EXPECTED_RESULT}"
        echo "Actual result: ${RESULT}"
        echo "Success"
        exit 0
    else
        echo "Expected result prefix: ${EXPECTED_RESULT}"
        echo "Actual result: ${RESULT}"
        echo "Failure; results not the same"
        exit 1
    fi
}

# Step 1. Build ExecuTorch and phi-3-mini runner
cmake_install_executorch_libraries
cmake_build_phi_3_mini

# Step 2. Export the tokenizer and model
prepare_tokenizer
export_phi_3_mini

# Step 3. Run and verify result
run_and_verify

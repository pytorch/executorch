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

  # Select workflow preset based on BUILD_TYPE
  if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    WORKFLOW_PRESET="llm-debug"
  else
    WORKFLOW_PRESET="llm-release"
  fi

  echo "Using workflow preset: ${WORKFLOW_PRESET}"
  cmake --workflow --preset ${WORKFLOW_PRESET}
}

cmake_build_phi_3_mini() {
  cmake -DCMAKE_PREFIX_PATH=${BUILD_DIR} \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -B${BUILD_DIR}/${MODEL_DIR} \
      ${MODEL_DIR}

  cmake --build ${BUILD_DIR}/${MODEL_DIR} -j${NPROC} --config ${BUILD_TYPE}
}

# Download tokenizer.model
prepare_tokenizer() {
  echo "Downloading tokenizer.model"
  wget -O tokenizer.model "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.model?download=true"
}

# Export phi-3-mini model to pte
export_phi_3_mini () {
  echo "Exporting phi-3-mini. This will take a few minutes"
  optimum-cli export executorch --model microsoft/Phi-3-mini-4k-instruct --task text-generation --recipe xnnpack --output_dir ./
}

run_and_verify() {
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run phi-3-mini runner at ${NOW}"
    if [[ ! -f "model.pte" ]]; then
        echo "Missing model artifact. Abort"
        exit 1
    fi
    if [[ ! -f "tokenizer.model" ]]; then
        echo "tokenizer.model is missing."
        exit 1
    fi

    ${BUILD_DIR}/${MODEL_DIR}/phi_3_mini_runner \
    --model_path=model.pte \
    --tokenizer_path=tokenizer.model \
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

# Step 2. Export the model
prepare_tokenizer
export_phi_3_mini

# Step 3. Run and verify result
run_and_verify

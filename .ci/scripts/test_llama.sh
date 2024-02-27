#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1 # stories110M.pt
BUILD_TOOL=$2 # buck2 or cmake
DTYPE=$3 # fp16 or fp32

if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit
fi

if [[ -z "${DTYPE:-}" ]]; then
  echo "Missing dtype, choose fp16 or fp32, exiting..."
  exit 1
fi

if [[ -z "${BUCK:-}" ]]; then
  BUCK=buck2
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"


cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out .
    cmake --build cmake-out -j9 --target install --config Release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    dir="examples/models/llama2"
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config Release

}

cleanup_files() {
  echo "Deleting downloaded and generated files"
  rm "${MODEL_NAME}"
  rm tokenizer.model
  rm tokenizer.bin
  rm "${EXPORTED_MODEL_NAME}"
  rm result.txt
  rm params.json
}

# Download and create artifacts.
PARAMS="params.json"
touch "${PARAMS}"
if [[ "${MODEL_NAME}" == "stories110M.pt" ]]; then
  download_stories_model_artifacts
else
  echo "Unsupported model name ${MODEL_NAME}"
  exit 1
fi

# Check dtype.
EXPORTED_MODEL_NAME="llama2"
if [[ "${DTYPE}" == "fp16" ]]; then
  EXPORTED_MODEL_NAME="${EXPORTED_MODEL_NAME}_h"
elif [[ "${DTYPE}" == "fp32" ]]; then
  :
else
  echo "Unsupported dtype ${DTYPE}"
  exit 1
fi

# Export model.
EXPORTED_MODEL_NAME="${EXPORTED_MODEL_NAME}.pte"
echo "Exporting ${EXPORTED_MODEL_NAME}"
$PYTHON_EXECUTABLE -m examples.models.llama2.export_llama -c stories110M.pt -p "${PARAMS}" -d "${DTYPE}"

# Create tokenizer.bin.
echo "Creating tokenizer.bin"
$PYTHON_EXECUTABLE -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin


RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --tokenizer_path=tokenizer.bin --prompt=Once --temperature=0 --seq_len=10"
# Check build tool.
echo "Running ${EXPORTED_MODEL_NAME} in portable mode"
if [[ "${BUILD_TOOL}" == "buck2" ]]; then
  # Run model.
  # shellcheck source=/dev/null
  $BUCK run examples/models/llama2:main -- ${RUNTIME_ARGS} > result.txt
elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
  cmake_install_executorch_libraries
  cmake_build_llama_runner
  # Run llama runner
  NOW=$(date +"%H:%M:%S")
  echo "Starting to run llama runner at ${NOW}"
  # shellcheck source=/dev/null
  cmake-out/examples/models/llama2/llama_main ${RUNTIME_ARGS} > result.txt
  NOW=$(date +"%H:%M:%S")
  echo "Finished at ${NOW}"
else
  echo "Invalid build tool ${BUILD_TOOL}. Only buck2 is supported atm"
  exit 1
fi
RESULT=$(cat result.txt)
# Check results.
EXPECTED_PREFIX="Once upon a time,"
# Expected result - may take too long to generate:
# "Once upon a time, there was a little girl named Lily. She loved to play outside" ...
if [[ "${RESULT}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Success"

  cleanup_files
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Failure; results not the same"

  cleanup_files
  exit 1
fi

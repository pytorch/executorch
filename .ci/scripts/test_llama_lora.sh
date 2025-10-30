#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake --preset llm \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build cmake-out -j9 --target install --config Release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    pushd extension/llm/tokenizers
    echo "Updating tokenizers submodule"
    git submodule update --init
    popd
    dir="examples/models/llama"
    retry cmake \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config Release
}

cleanup_files() {
  echo "Deleting downloaded and generated files"
  rm -rf "${DOWNLOADED_PATH}/"
  rm result.txt
}

# Download model artifacts from HF Hub.
# Hosting in personal repo for now.
HF_MODEL_REPO="lucylq/llama3_1B_lora"
DOWNLOADED_PATH=$(
  bash "$(dirname "${BASH_SOURCE[0]}")/download_hf_hub.sh" \
    --model_id "${HF_MODEL_REPO}" \
    --files "adapter_config.json" "adapter_model.pt" "consolidated.00.pth" "params.json" "tokenizer.model"
)
# Build llama runner.
cmake_install_executorch_libraries
cmake_build_llama_runner

# Constants.
RUNTIME_ARGS="--tokenizer_path=${DOWNLOADED_PATH}/tokenizer.model --temperature=0 --seq_len=20 --warmup=1"
PROMPT="What happens if you eat watermelon seeds?"
EXPECTED_PREFIX="What happens if you eat watermelon seeds? Watermelon seeds are a good source of vitamin C and"

# Export LoRA PTE file.
MODEL_NAME="llama_3_2_1B_lora"
$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    base.checkpoint="${DOWNLOADED_PATH}/consolidated.00.pth" \
    base.params="${DOWNLOADED_PATH}/params.json" \
    base.adapter_checkpoint="${DOWNLOADED_PATH}/adapter_model.pt" \
    base.adapter_config="${DOWNLOADED_PATH}/adapter_config.json" \
    base.tokenizer_path="${DOWNLOADED_PATH}/tokenizer.model" \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.dtype_override="fp32" \
    backend.xnnpack.enabled=true \
    backend.xnnpack.extended_ops=true \
    export.output_name="${MODEL_NAME}.pte"

# Run llama runner
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --model_path=${MODEL_NAME}.pte --prompt="${PROMPT}" ${RUNTIME_ARGS} > result.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result.txt)
if [[ "${RESULT}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  # Do not clean up files if test passes, as they're re-used in the next test.
  echo "Success"
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Failure; results not the same"
  cleanup_files
  exit 1
fi

# Export LoRA PTE, foundation PTD file.
MODEL_SEPARATE="${MODEL_NAME}_separate"
$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    base.checkpoint="${DOWNLOADED_PATH}/consolidated.00.pth" \
    base.params="${DOWNLOADED_PATH}/params.json" \
    base.adapter_checkpoint="${DOWNLOADED_PATH}/adapter_model.pt" \
    base.adapter_config="${DOWNLOADED_PATH}/adapter_config.json" \
    base.tokenizer_path="${DOWNLOADED_PATH}/tokenizer.model" \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.dtype_override="fp32" \
    backend.xnnpack.enabled=true \
    backend.xnnpack.extended_ops=true \
    export.output_name="${MODEL_SEPARATE}.pte" \
    export.foundation_weights_file="${MODEL_SEPARATE}.ptd"

# Run llama runner.
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --model_path=${MODEL_SEPARATE}.pte --data_paths=${MODEL_SEPARATE}.ptd --prompt="${PROMPT}" ${RUNTIME_ARGS} > result2.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT2=$(cat result2.txt)
if [[ "${RESULT2}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT2}"
  # Do not clean up files if test passes, as they're re-used in the next test.
  echo "Success"
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT2}"
  echo "Failure; results not the same"
  cleanup_files
  exit 1
fi

# Export LoRA PTE, LoRA PTD, foundation PTD file.
MODEL_PROGRAM_ONLY="${MODEL_NAME}_program"
MODEL_LORA_WEIGHTS="lora_weights"
MODEL_FOUNDATION_WEIGHTS="foundation_weights"
$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    base.checkpoint="${DOWNLOADED_PATH}/consolidated.00.pth" \
    base.params="${DOWNLOADED_PATH}/params.json" \
    base.adapter_checkpoint="${DOWNLOADED_PATH}/adapter_model.pt" \
    base.adapter_config="${DOWNLOADED_PATH}/adapter_config.json" \
    base.tokenizer_path="${DOWNLOADED_PATH}/tokenizer.model" \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.dtype_override="fp32" \
    backend.xnnpack.enabled=true \
    backend.xnnpack.extended_ops=true \
    export.output_name="${MODEL_PROGRAM_ONLY}.pte" \
    export.foundation_weights_file="${MODEL_FOUNDATION_WEIGHTS}.ptd" \
    export.lora_weights_file="${MODEL_LORA_WEIGHTS}.ptd"

# Run llama runner.
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --model_path=${MODEL_PROGRAM_ONLY}.pte --data_paths="${MODEL_FOUNDATION_WEIGHTS}.ptd,${MODEL_LORA_WEIGHTS}.ptd" --prompt="${PROMPT}" ${RUNTIME_ARGS} > result3.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT3=$(cat result3.txt)
if [[ "${RESULT3}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT3}"
  echo "Success"
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT3}"
  echo "Failure; results not the same"
  cleanup_files
  exit 1
fi

cleanup_files

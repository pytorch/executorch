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
EXPORTED_MODEL_NAME="llama_3_2_1B_lora.pte"
# Export model.
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
    export.output_name="${EXPORTED_MODEL_NAME}"

# Build llama runner.
cmake_install_executorch_libraries
cmake_build_llama_runner

PROMPT="What happens if you eat watermelon seeds?"
# Run llama runner
RUNTIME_ARGS="--model_path=${EXPORTED_MODEL_NAME} --tokenizer_path=${DOWNLOADED_PATH}/tokenizer.model --temperature=0 --seq_len=20 --warmup=1"

NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --prompt="${PROMPT}" ${RUNTIME_ARGS} > result.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result.txt)
EXPECTED_PREFIX="What happens if you eat watermelon seeds? Watermelon seeds are a good source of vitamin C,"

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

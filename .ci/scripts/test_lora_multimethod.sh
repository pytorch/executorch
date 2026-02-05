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
    cmake --workflow llm-release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    pushd extension/llm/tokenizers
    echo "Updating tokenizers submodule"
    git submodule update --init
    popd
    make llama-cpu
}

cleanup_files() {
  echo "Deleting downloaded and generated files"
  rm -rf "${HF_QWEN_PATH}/"
  rm -rf "${HF_ADAPTER_PATH}/"
  rm -rf *.pte
  rm -f result*.txt
}

# Download LoRA adapter.
python -m pip install -q huggingface_hub
HF_ADAPTER_REPO="lucylq/qwen3_06B_lora_math"
HF_ADAPTER_PATH=$(
  bash "$(dirname "${BASH_SOURCE[0]}")/download_hf_hub.sh" \
    --model_id "${HF_ADAPTER_REPO}" \
    --files "adapter_config.json" "adapter_model.safetensors"
)

# Download base model (for tokenizer path).
HF_QWEN_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Qwen3-0.6B'))")
echo "Model downloaded to: $HF_QWEN_PATH"

### EXPORT MULTIMETHOD PTE ###
# Set environment variables for OmegaConf interpolation in yaml.
export LORA_ADAPTER_CHECKPOINT="${HF_ADAPTER_PATH}/adapter_model.safetensors"
export LORA_ADAPTER_CONFIG="${HF_ADAPTER_PATH}/adapter_config.json"

$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    --config examples/models/qwen3/config/qwen3_multimethod.yaml

### BUILD LLAMA RUNNER ###
cmake_install_executorch_libraries
cmake_build_llama_runner

# Runner constants.
RUNTIME_ARGS="--tokenizer_path=${HF_QWEN_PATH}/ --temperature=0 --seq_len=100 --warmup=1"
PROMPT="<|im_start|>user Calculate 15% of 80?<|im_end|><|im_start|>assistant"

# Expected outputs.
EXPECTED_LORA_PREFIX="
<|im_start|>user Calculate 15% of 80?<|im_end|><|im_start|>assistant
To calculate 15% of 80"

EXPECTED_BASE_PREFIX="<|im_start|>user Calculate 15% of 80?<|im_end|><|im_start|>assistant:
<think>
Okay, so I need to calculate 15% of 80."

### TEST 1: Run lora_forward method ###
NOW=$(date +"%H:%M:%S")
echo "Test 1: Multimethod lora_forward. Starting at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main \
    --model_path=multimethod_qwen.pte \
    --method_name=lora_forward \
    --prompt="${PROMPT}" \
    ${RUNTIME_ARGS} > result_lora.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result_lora.txt)
if [[ "${RESULT}" == "${EXPECTED_LORA_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_LORA_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Test 1 (lora_forward): Success"
else
  echo "Expected result prefix: ${EXPECTED_LORA_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Test 1 (lora_forward): Failure"
  cleanup_files
  exit 1
fi

### TEST 2: Run base_forward method ###
NOW=$(date +"%H:%M:%S")
echo "Test 2: Multimethod base_forward. Starting at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main \
    --model_path=multimethod_qwen.pte \
    --method_name=base_forward \
    --prompt="${PROMPT}" \
    ${RUNTIME_ARGS} > result_base.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT=$(cat result_base.txt)
if [[ "${RESULT}" == "${EXPECTED_BASE_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_BASE_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Test 2 (base_forward): Success"
else
  echo "Expected result prefix: ${EXPECTED_BASE_PREFIX}"
  echo "Actual result: ${RESULT}"
  echo "Test 2 (base_forward): Failure"
  cleanup_files
  exit 1
fi

echo "Multimethod tests passed!"
cleanup_files

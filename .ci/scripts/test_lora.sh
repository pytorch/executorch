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
  rm -rf *.pte *.ptd
  rm result*.txt
}

# Hosting lora adapter in personal repo for now.
python -m pip install -q huggingface_hub
HF_ADAPTER_REPO="lucylq/qwen3_06B_lora_math"
HF_ADAPTER_PATH=$(
  bash "$(dirname "${BASH_SOURCE[0]}")/download_hf_hub.sh" \
    --model_id "${HF_ADAPTER_REPO}" \
    --files "adapter_config.json" "adapter_model.safetensors"
)

### SINGLE LORA PTE ###
# Export LoRA PTE file.
$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    --config examples/models/qwen3/config/qwen3_xnnpack.yaml \
    +base.adapter_checkpoint="${HF_ADAPTER_PATH}/adapter_model.safetensors" \
    +base.adapter_config="${HF_ADAPTER_PATH}/adapter_config.json" \
    +export.output_name="qwen_lora_math_full.pte"

# Capture the path of the downloaded qwen artifacts
HF_QWEN_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Qwen3-0.6B'))")
echo "Model downloaded to: $HF_QWEN_PATH"

### BUILD LLAMA RUNNER.
cmake_install_executorch_libraries
cmake_build_llama_runner

# Runner constants.
RUNTIME_ARGS="--tokenizer_path=${HF_QWEN_PATH}/ --temperature=0 --seq_len=100 --warmup=1"
PROMPT="<|im_start|>user Calculate 15% of 80?<|im_end|><|im_start|>assistant"
EXPECTED_PREFIX="
<|im_start|>user Calculate 15% of 80?<|im_end|><|im_start|>assistant
To calculate 15% of 80, we can multiply 80 by 0.15.
80 * 0.15 = 12
So, 15% of 80 is 12.
#### 12
The answer is: 12<|im_end|>"

# Run llama runner on single lora PTE file.
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --model_path=qwen_lora_math_full.pte --prompt="${PROMPT}" ${RUNTIME_ARGS} > result.txt
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

### PROGRAM DATA SEPARATION ###
# Export LoRA PTE, LoRA PTD, foundation PTD file.
$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    --config examples/models/qwen3/config/qwen3_xnnpack.yaml \
    +base.adapter_checkpoint="${HF_ADAPTER_PATH}/adapter_model.safetensors" \
    +base.adapter_config="${HF_ADAPTER_PATH}/adapter_config.json" \
    +export.output_name="qwen_lora_math.pte" \
    +export.foundation_weights_file="qwen_foundation.ptd" \
    +export.lora_weights_file="qwen_lora_math.ptd"

# Run llama runner on PTE, PTD files.
NOW=$(date +"%H:%M:%S")
echo "Starting to run llama runner at ${NOW}"
# shellcheck source=/dev/null
cmake-out/examples/models/llama/llama_main --model_path=qwen_lora_math.pte --data_paths="qwen_foundation.ptd,qwen_lora_math.ptd" --prompt="${PROMPT}" ${RUNTIME_ARGS} > result2.txt
NOW=$(date +"%H:%M:%S")
echo "Finished at ${NOW}"

RESULT2=$(cat result2.txt)
if [[ "${RESULT2}" == "${EXPECTED_PREFIX}"* ]]; then
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT2}"
  echo "Success"
else
  echo "Expected result prefix: ${EXPECTED_PREFIX}"
  echo "Actual result: ${RESULT2}"
  echo "Failure; results not the same"
  cleanup_files
  exit 1
fi

cleanup_files

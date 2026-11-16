#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# Disable HF Xet storage to avoid stalled downloads on CI runners
export HF_HUB_DISABLE_XET=1
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    cmake --preset llm-release -DEXECUTORCH_ENABLE_LOGGING=ON
    cmake --build --preset llm-release-install
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
  # Only what this script generated. HF_QWEN_PATH and HF_ADAPTER_PATH point
  # inside the huggingface_hub cache: on OSDC that is a shared read-only mount,
  # so removing them failed the job after the tests had already passed, and
  # anywhere else it discards a cache entry the next job wants. A teardown also
  # must not fail a run whose tests passed.
  echo "Deleting generated files"
  rm -rf ./*.pte || true
  rm -f result*.txt || true
}

matches_base_response_prefix() {
  local output_file="$1"
  python - "$output_file" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
# Under 8da4w, the base model's first token is a near-tie between ":\n" and ":"
# that tips with small numeric changes, such as RMSNorm and QK-norm moving into
# XNNPACK. Accept both observed continuations.
pattern = re.compile(
    r"^<\|im_start\|>user Calculate 15% of 80\?<\|im_end\|><\|im_start\|>assistant:"
    r"(?:\n(?:<think>\n)+| me\n)"
    r"Okay, so I need to calculate 15% of 80\.",
    re.MULTILINE,
)
if not pattern.match(text):
    print(f"Expected a match for: {pattern.pattern}")
    sys.exit(1)
PY
}

# Download LoRA adapter.
# The Hugging Face downloads below are pinned to commits the read-only
# CI cache holds. To bump a pin, seed the cache first from a pytorch/executorch
# branch (forks cannot write the cache): run pull.yml on it with
# workflow_dispatch, or push to it after its pull request has the
# ci-refresh-hf-cache label.
python -m pip install -q huggingface_hub
HF_ADAPTER_REPO="lucylq/qwen3_06B_lora_math"
HF_ADAPTER_PATH=$(
  bash "$(dirname "${BASH_SOURCE[0]}")/download_hf_hub.sh" \
    --model_id "${HF_ADAPTER_REPO}" \
    --revision 0e6bc3a88a597c5abc4872734eec8db3a9fdd846 \
    --files "adapter_config.json" "adapter_model.safetensors"
)

# Download base model (for tokenizer path).
HF_QWEN_PATH=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Qwen3-0.6B', revision='db4e90b8359f816f09bf5c944797ce116f26da3c'))")
echo "Model downloaded to: $HF_QWEN_PATH"

# Without a checkpoint, export downloads Qwen/Qwen3-0.6B from its main branch,
# so convert the pinned commit the read-only CI cache holds instead.
QWEN_SNAPSHOT=$($PYTHON_EXECUTABLE -m executorch.examples.models.llama.hf_download \
  snapshot Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca)
QWEN_CHECKPOINT_DIR=$(mktemp -d "${RUNNER_TEMP:-/tmp}/qwen3_0_6b_XXXXXX")
trap 'rm -rf "${QWEN_CHECKPOINT_DIR}"' EXIT
QWEN_CHECKPOINT="${QWEN_CHECKPOINT_DIR}/qwen3_0_6b.pth"
$PYTHON_EXECUTABLE -m executorch.examples.models.qwen3.convert_weights \
  "${QWEN_SNAPSHOT}" "${QWEN_CHECKPOINT}"

### EXPORT MULTIMETHOD PTE ###
# Set environment variables for OmegaConf interpolation in yaml.
export LORA_ADAPTER_CHECKPOINT="${HF_ADAPTER_PATH}/adapter_model.safetensors"
export LORA_ADAPTER_CONFIG="${HF_ADAPTER_PATH}/adapter_config.json"

$PYTHON_EXECUTABLE -m extension.llm.export.export_llm \
    --config examples/models/qwen3/config/qwen3_multimethod.yaml \
    +base.checkpoint="${QWEN_CHECKPOINT}"

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
  echo "Test 1 (lora_forward): Success"
else
  echo "Test 1 (lora_forward): Failure"
  echo "Expected result prefix: ${EXPECTED_LORA_PREFIX}"
  echo "Actual result: ${RESULT}"
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
if matches_base_response_prefix result_base.txt; then
  echo "Test 2 (base_forward): Success"
else
  echo "Test 2 (base_forward): Failure"
  echo "Actual result: ${RESULT}"
  cleanup_files
  exit 1
fi

echo "Multimethod tests passed!"
cleanup_files

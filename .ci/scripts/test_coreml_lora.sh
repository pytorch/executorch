#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
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

EXPORT_SCRIPT="examples/apple/coreml/llama/export_static_llm_coreml.py"
RUN_SCRIPT="examples/apple/coreml/llama/run_static_llm.py"
RUN_MF_SCRIPT="examples/apple/coreml/llama/run_static_llm_multifunction.py"

# Export parameters — small context for fast CI.
MAX_CONTEXT_LEN=64
INPUT_LEN=32
CACHE_LEN=$((MAX_CONTEXT_LEN - INPUT_LEN))

cleanup_files() {
  echo "Deleting generated files"
  rm -f base.pte lora.pte lora_mf.pte
  rm -f result_base*.txt result_lora*.txt
  rm -rf "${ADAPTER_DIR}"
}

### SETUP ###
pushd "${EXECUTORCH_ROOT}/examples/apple/coreml/llama"

# Download stories110M artifacts.
download_stories_model_artifacts

# Create a synthetic LoRA adapter for stories110M.
ADAPTER_DIR=$(mktemp -d)
${PYTHON_EXECUTABLE} - "${ADAPTER_DIR}" <<'PYEOF'
import json
import sys
import torch
from safetensors.torch import save_file

adapter_dir = sys.argv[1]
dim = 768
n_heads = 12
n_layers = 12
rank = 8
alpha = 16
target_modules = ["q_proj", "v_proj"]

config = {
    "r": rank,
    "lora_alpha": alpha,
    "target_modules": target_modules,
}
with open(f"{adapter_dir}/adapter_config.json", "w") as f:
    json.dump(config, f)

# Create adapter weights in unsloth format.
# lora_A: [rank, in_features], lora_B: [out_features, rank]
# Initialize lora_B to zeros so the adapter is initially a no-op,
# meaning base and lora outputs should match.
tensors = {}
for i in range(n_layers):
    for proj in target_modules:
        prefix = f"base_model.model.model.layers.{i}.self_attn.{proj}"
        tensors[f"{prefix}.lora_A.weight"] = torch.randn(rank, dim) * 0.01
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(dim, rank)

save_file(tensors, f"{adapter_dir}/adapter_model.safetensors")
print(f"Created synthetic adapter in {adapter_dir}")
PYEOF

ADAPTER_CHECKPOINT="${ADAPTER_DIR}/adapter_model.safetensors"
ADAPTER_CONFIG="${ADAPTER_DIR}/adapter_config.json"

popd

### TEST 1: Base only (single method) ###
echo "=== Test 1: Base only (single method) ==="
${PYTHON_EXECUTABLE} "${EXPORT_SCRIPT}" \
    --checkpoint examples/apple/coreml/llama/stories110M.pt \
    --params examples/apple/coreml/llama/params.json \
    --output base.pte \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --input_len ${INPUT_LEN}

BASE_SIZE=$(stat -f%z base.pte 2>/dev/null || stat -c%s base.pte)
echo "Test 1: base.pte size = ${BASE_SIZE} bytes"

### TEST 2: Base + LoRA adapter (multimethod, no multifunction) ###
echo "=== Test 2: Base + LoRA adapter ==="
${PYTHON_EXECUTABLE} "${EXPORT_SCRIPT}" \
    --checkpoint examples/apple/coreml/llama/stories110M.pt \
    --params examples/apple/coreml/llama/params.json \
    --output lora.pte \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --input_len ${INPUT_LEN} \
    --adapter lora "${ADAPTER_CHECKPOINT}" "${ADAPTER_CONFIG}"

LORA_SIZE=$(stat -f%z lora.pte 2>/dev/null || stat -c%s lora.pte)
echo "Test 2: lora.pte size = ${LORA_SIZE} bytes"

### TEST 3: Base + LoRA + multifunction ###
echo "=== Test 3: Base + LoRA + multifunction ==="
${PYTHON_EXECUTABLE} "${EXPORT_SCRIPT}" \
    --checkpoint examples/apple/coreml/llama/stories110M.pt \
    --params examples/apple/coreml/llama/params.json \
    --output lora_mf.pte \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --input_len ${INPUT_LEN} \
    --multifunction \
    --adapter lora "${ADAPTER_CHECKPOINT}" "${ADAPTER_CONFIG}"

LORA_MF_SIZE=$(stat -f%z lora_mf.pte 2>/dev/null || stat -c%s lora_mf.pte)
echo "Test 3: lora_mf.pte size = ${LORA_MF_SIZE} bytes"

### FILE SIZE CHECKS ###
echo ""
echo "=== File size summary ==="
echo "  Base:                ${BASE_SIZE} bytes"
echo "  Base + LoRA:         ${LORA_SIZE} bytes"
echo "  Base + LoRA + MF:    ${LORA_MF_SIZE} bytes"

# LoRA PTE size should be close to base size.
# skip_split_names prevents splitting LoRA-targeted modules (for POSITIONAL
# weight sharing), so lora.pte may be slightly smaller than base.pte.
LORA_DIFF=$((LORA_SIZE - BASE_SIZE))
if [[ ${LORA_DIFF} -lt 0 ]]; then
  ABS_LORA_DIFF=$((-LORA_DIFF))
else
  ABS_LORA_DIFF=${LORA_DIFF}
fi
echo "  LoRA size difference: ${LORA_DIFF} bytes"

MAX_LORA_DIFF=$((BASE_SIZE / 10))
if [[ ${ABS_LORA_DIFF} -gt ${MAX_LORA_DIFF} ]]; then
  echo "FAIL: LoRA size difference ${LORA_DIFF} exceeds 10% of base size ${BASE_SIZE}"
  cleanup_files
  exit 1
fi

# Multifunction PTE should be close to LoRA PTE size.
# POSITIONAL sharing deduplicates base weights across methods.
MF_DIFF=$((LORA_MF_SIZE - LORA_SIZE))
if [[ ${MF_DIFF} -lt 0 ]]; then
  ABS_MF_DIFF=$((-MF_DIFF))
else
  ABS_MF_DIFF=${MF_DIFF}
fi
echo "  Multifunction difference: ${MF_DIFF} bytes"

MAX_MF_DIFF=$((BASE_SIZE / 20))
if [[ ${ABS_MF_DIFF} -gt ${MAX_MF_DIFF} ]]; then
  echo "FAIL: Multifunction difference ${MF_DIFF} exceeds 5% of base size ${BASE_SIZE}"
  cleanup_files
  exit 1
fi

echo "File size checks passed."

### INFERENCE TESTS ###
# These require CoreML runtime (macOS with ANE).
# Skip if not on macOS or if explicitly disabled.
if [[ "$(uname)" != "Darwin" ]] || [[ "${SKIP_INFERENCE:-0}" == "1" ]]; then
  echo "Skipping inference tests (not on macOS or SKIP_INFERENCE=1)"
  cleanup_files
  exit 0
fi

RUNNER_ARGS="--params examples/apple/coreml/llama/params.json --tokenizer examples/apple/coreml/llama/tokenizer.model --temperature 0 --max_new_tokens 20 --input_len ${INPUT_LEN} --cache_len ${CACHE_LEN}"
PROMPT="Once upon a time,"

# Test 1 inference: base only
echo ""
echo "=== Test 1 inference: base (single method) ==="
${PYTHON_EXECUTABLE} "${RUN_SCRIPT}" \
    --model base.pte \
    --prompt "${PROMPT}" \
    ${RUNNER_ARGS} > result_base.txt 2>&1 || true
echo "Base output:"
cat result_base.txt

# Test 2 inference: base method from lora PTE
echo ""
echo "=== Test 2 inference: base method (from lora PTE) ==="
# The base method is "forward" in the multimethod PTE.
${PYTHON_EXECUTABLE} "${RUN_SCRIPT}" \
    --model lora.pte \
    --prompt "${PROMPT}" \
    ${RUNNER_ARGS} > result_lora_base.txt 2>&1 || true
echo "LoRA PTE base output:"
cat result_lora_base.txt

# Test 2 inference: lora method from lora PTE
echo ""
echo "=== Test 2 inference: lora method (from lora PTE) ==="
${PYTHON_EXECUTABLE} "${RUN_SCRIPT}" \
    --model lora.pte \
    --method lora \
    --prompt "${PROMPT}" \
    ${RUNNER_ARGS} > result_lora_lora.txt 2>&1 || true
echo "LoRA PTE lora output:"
cat result_lora_lora.txt

# Test 3 inference: multifunction lora PTE
echo ""
echo "=== Test 3 inference: multifunction ==="
${PYTHON_EXECUTABLE} "${RUN_MF_SCRIPT}" \
    --model lora_mf.pte \
    --prompt "${PROMPT}" \
    --max_context_len ${MAX_CONTEXT_LEN} \
    --max_new_tokens 20 \
    --temperature 0 \
    --params examples/apple/coreml/llama/params.json \
    --tokenizer examples/apple/coreml/llama/tokenizer.model > result_lora_mf.txt 2>&1 || true
echo "Multifunction output:"
cat result_lora_mf.txt

# Since lora_B is initialized to zeros, the LoRA adapter is a no-op.
# Base output from Test 1 and LoRA output from Test 2 should match.
echo ""
echo "=== Output comparison ==="
echo "Base and LoRA outputs should match (zero adapter)."

echo ""
echo "All CoreML LoRA export tests passed!"
cleanup_files

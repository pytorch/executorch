#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
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

# Update tokenizers submodule
pushd $EXECUTORCH_ROOT/extension/llm/tokenizers
echo "Update tokenizers submodule"
git submodule update --init
popd

pushd $EXECUTORCH_ROOT/examples/apple/coreml/llama

# Download stories llama110m artifacts
download_stories_model_artifacts

# Test static ANE llama model export
echo "Exporting static ANE llama model..."
python export_static_llm_coreml.py --checkpoint stories110M.pt --params params.json --output model.pte

# The ANE is not accessible in github CI, so we export with CPU to test runner
echo "Exporting CPU-only model for CI testing..."
python export_static_llm_coreml.py --checkpoint stories110M.pt --params params.json --output model_cpu.pte --cpu_only

popd

# Build the C++ runner
echo "Building C++ runner..."
BUILD_DIR="${EXECUTORCH_ROOT}/cmake-out"

# Clean build directory completely to avoid stale artifacts and generator conflicts
rm -rf "${BUILD_DIR}"

cmake -S "${EXECUTORCH_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_COREML=ON \
  -G Ninja

cmake --build "${BUILD_DIR}" -j --target run_static_llm_coreml --config Release

# TODO: enable runner once CoreML bug with caching is fixed
# # Run the C++ runner with the CPU model
# echo "Running C++ runner with CPU model..."
# RUNNER="${BUILD_DIR}/examples/apple/coreml/llama/runner/run_static_llm_coreml"
# MODEL_DIR="${EXECUTORCH_ROOT}/examples/apple/coreml/llama"

# # Run the model and capture full output for debugging
# FULL_OUTPUT=$("${RUNNER}" \
#   --model "${MODEL_DIR}/model.pte" \
#   --params "${MODEL_DIR}/params.json" \
#   --tokenizer "${MODEL_DIR}/tokenizer.model" \
#   --prompt "Once upon a time," \
#   --max_new_tokens 50 2>&1)

# echo "Full output:"
# echo "${FULL_OUTPUT}"

# # Check that the model produced meaningful output
# # The output should contain: the prompt "Once upon a time," and the continuation including "there was"
# # Due to log interleaving, we check for individual key parts separately
# if [[ "${FULL_OUTPUT}" == *"Once upon a time,"* ]] && [[ "${FULL_OUTPUT}" == *"there"* ]] && [[ "${FULL_OUTPUT}" == *"was"* ]]; then
#   echo "Output contains expected prompt and generated text"
#   echo "C++ runner test passed!"
# else
#   echo "ERROR: Output does not contain expected text"
#   echo "Expected: 'Once upon a time,' followed by 'there' and 'was'"
#   exit 1
# fi

# TODO: enable runner once CoreML bug with caching is fixed
# # Run lookahead decoding test (currently produces <unk> tokens on stories, but works with llama)
# echo "Running C++ runner with lookahead decoding..."
# "${RUNNER}" \
#   --model "${MODEL_DIR}/model.pte" \
#   --params "${MODEL_DIR}/params.json" \
#   --tokenizer "${MODEL_DIR}/tokenizer.model" \
#   --prompt "Once upon a time," \
#   --max_new_tokens 50 \
#   --lookahead

# Test export of deprecated model
pushd $EXECUTORCH_ROOT/examples/apple/coreml/llama
python export.py -n model.pte -p params.json -c stories110M.pt --seq_length 32 --max_seq_length 64 --dtype fp16 --coreml-quantize c4w --embedding-quantize 4,32
popd

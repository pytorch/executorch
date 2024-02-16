#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

MODEL_NAME=$1 # stories110M.pt
BUILD_TOOL=$2 # buck2
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

which "${PYTHON_EXECUTABLE}"

# Check build tool.
if [[ "${BUILD_TOOL}" == "buck2" ]]; then
  :
else
  echo "Invalid build tool ${BUILD_TOOL}. Only buck2 is supported atm"
  exit 1
fi

cleanup_files() {
  echo "Deleting downloaded and generated files"
  rm "${MODEL_NAME}"
  rm tokenizer.model
  rm tokenizer.bin
  rm "${EXPORTED_MODEL_NAME}"
}

# Download and create artifacts.
PARAMS="params.json"
touch "${PARAMS}"
if [[ "${MODEL_NAME}" == "stories110M.pt" ]]; then
  # Download stories110M.pt and tokenizer from Github
  wget "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt"
  wget "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"
  # Create params.json file
  echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' > "${PARAMS}"
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
python3 -m examples.models.llama2.export_llama -c stories110M.pt -p "${PARAMS}" -d "${DTYPE}"

# Create tokenizer.bin.
echo "Creating tokenizer.bin"
buck2 run examples/models/llama2/tokenizer:tokenizer_py -- -t tokenizer.model -o tokenizer.bin

# Run model.
echo "Running ${EXPORTED_MODEL_NAME} in portable mode"
RESULT=$(timeout 500s buck2 run examples/models/llama2:main -- --model_path="${EXPORTED_MODEL_NAME}" --tokenizer_path=tokenizer.bin --prompt="Once" --temperature=0) || true

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

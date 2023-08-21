#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

MODEL_NAME=$1
if [[ -z "${MODEL_NAME:-}" ]]; then
  echo "Missing model name, exiting..."
  exit 1
fi

BUILD_TOOL=$2
if [[ -z "${BUILD_TOOL:-}" ]]; then
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

echo "Testing ${MODEL_NAME} with ${BUILD_TOOL}..."

test_model() {
  python -m examples.export.export_example --model_name="${MODEL_NAME}"

  # Run test model
  if [[ $1 == "buck2" ]]; then
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
  elif [[ $1 == "cmake" ]]; then
    ./"${CMAKE_OUTPUT_DIR}"/executor_runner --model_path "./${MODEL_NAME}.pte"
  else
    echo "Invalid build tool $1. Only buck2 and cmake are supported atm"
    exit 1
  fi
}

test_quantized_model() {
  python -m examples.quantization.example --model_name="${MODEL_NAME}"
}

which python
# Test the select model
test_model
test_quantized_model

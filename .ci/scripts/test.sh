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

QUANTIZATION=$3
if [[ -z "${QUANTIZATION:-}" ]]; then
  QUANTIZATION=false
fi

test_model() {
  python -m examples.export.export_example --model_name="${MODEL_NAME}"

  # Run test model
  if [[ "${BUILD_TOOL}" == "buck2" ]]; then
    buck2 run //examples/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
  elif [[ "${BUILD_TOOL}" == "cmake" ]]; then
    CMAKE_OUTPUT_DIR=cmake-out
    ./${CMAKE_OUTPUT_DIR}/executor_runner --model_path "./${MODEL_NAME}.pte"
  else
    echo "Invalid build tool ${BUILD_TOOL}. Only buck2 and cmake are supported atm"
    exit 1
  fi
}


which python

echo "Testing ${MODEL_NAME} with ${BUILD_TOOL}..."
# Test the select model
test_model

if [[ "${QUANTIZATION}" == true ]]; then
  bash examples/quantization/test_quantize.sh "${MODEL_NAME}"
else
  echo "The model ${MODEL_NAME} doesn't support quantization yet"
fi

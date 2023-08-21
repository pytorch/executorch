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
else
  echo "Testing ${MODEL_NAME} ..."
fi

test_model() {
  python -m examples.export.export_example --model_name="${MODEL_NAME}"

  # Run test model
  ./"${CMAKE_OUTPUT_DIR}"/executor_runner --model_path "./${MODEL_NAME}.pte"
}

build_and_test_executorch() {
  build_executorch_runner cmake

  which python
  # Test the select model
  test_model
}

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

install_executorch
build_and_test_executorch

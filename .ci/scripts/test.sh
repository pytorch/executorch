#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

test_model() {
  MODEL_NAME=$1
  python -m examples.export.export_example --model_name="${MODEL_NAME}"

  # Run test model
  buck2 run //examples/executor_runner:executor_runner -- --model_path "./${MODEL_NAME}.pte"
}

build_and_test_executorch() {
  # Build executorch runtime
  buck2 build //examples/executor_runner:executor_runner

  which python
  # Test the example linear model
  test_model "linear"
}

echo ">>> ${MODEL_NAME} <<<"

install_executorch
build_and_test_executorch

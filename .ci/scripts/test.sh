#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

install_executorch() {
  which pip
  # Install executorch, this assumes that Executorch is checked out in the
  # current directory
  pip install .
  # Just print out the list of packages for debugging
  pip list
}

build_and_test_executorch() {
  # Build executorch runtime
  buck2 build //examples/executor_runner:executor_runner

  which python
  # Export a test model
  python -m examples.export.export_example --model_name="linear"
  # Run test model
  buck2 run //examples/executor_runner:executor_runner -- --model_path ./linear.pte

  # Export delegate models
  python -m examples.export.export_and_delegate --option "partition"

  # Run delegate models
  buck2 run //examples/executor_runner:executor_runner -- --model_path ./partition_lowered_model.pte
}

install_executorch
build_and_test_executorch

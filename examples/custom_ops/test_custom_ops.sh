#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of using custom operator in a PyTorch model and use EXIR to capture and export a model file. Then use `executor_runner` demo C++ binary to run the model.

test_custom_op_1() {
  echo 'Exporting custom_ops_1.pte'
  python3 -m examples.custom_ops.custom_ops_1
  # should save file custom_ops_1.pte

  echo 'Running executor_runner'
  buck2 run //fbcode/executorch/examples/executor_runner:executor_runner -- --model_path=./custom_ops_1.pte
  # should give correct result

  echo 'Removing custom_ops_1.pte'
  rm ./custom_ops_1.pte
}

test_custom_op_1

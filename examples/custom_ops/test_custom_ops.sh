#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of using custom operator in a PyTorch model and use
# EXIR to capture and export a model file. Then use `executor_runner` demo C++
# binary to run the model.

set -e

test_buck2_custom_op_1() {
  local model_name='custom_ops_1'
  echo "Exporting ${model_name}.pte"
  python3 -m "examples.custom_ops.${model_name}"
  # should save file custom_ops_1.pte

  echo 'Running executor_runner'
  buck2 run //examples/executor_runner:executor_runner \
      --config=executorch.register_custom_op_1=1 -- --model_path="./${model_name}.pte"
  # should give correct result

  echo "Removing ${model_name}.pte"
  rm "./${model_name}.pte"
}

test_cmake_custom_op_1() {
  local model_name='custom_ops_1'
  echo "Exporting ${model_name}.pte"
  python3 -m "examples.custom_ops.${model_name}"
  # should save file custom_ops_1.pte
  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DBUCK2=buck2 -DREGISTER_EXAMPLE_CUSTOM_OP_1=ON ..)

  echo 'Building executor_runner'
  cmake --build cmake-out -j9

  echo 'Running executor_runner'
  cmake-out/executor_runner --model_path="./${model_name}.pte"
}

test_cmake_custom_op_2() {
  local model_name='custom_ops_2'
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"

  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DBUCK2=buck2 \
      -DREGISTER_EXAMPLE_CUSTOM_OP_2=ON \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" ..)

  echo 'Building executor_runner'
  cmake --build cmake-out -j9

  echo "Exporting ${model_name}.pte"
  python3 -m "examples.custom_ops.${model_name}"
  # should save file custom_ops_2.pte

  echo 'Running executor_runner'
  cmake-out/executor_runner "--model_path=./${model_name}.pte"
}

# DEBUG
which python3
python3 -mpip list

test_buck2_custom_op_1
test_cmake_custom_op_1
test_cmake_custom_op_2

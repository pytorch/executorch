#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of selective build, using 3 APIs:
# 1. Select all ops
# 2. Select from a list of ops
# 3. Select from a yaml file
# 4. (TODO) Select from a serialized model (.pte)
set -e

test_buck2_select_all_ops() {
    echo "Exporting MobilenetV3"
    ${PYTHON_EXECUTABLE} -m examples.export.export_example --model_name="mv3"

    echo "Running executor_runner"
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=all -- --model_path=./mv3.pte

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_buck2_select_ops_in_list() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.export.export_example --model_name="add_mul"

    echo "Running executor_runner"
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=list -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_from_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.custom_ops.custom_ops_1

    echo "Running executor_runner"
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=yaml -- --model_path=./custom_ops_1.pte

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

# TODO(larryliu0820) Add example to select ops from model.

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi


test_buck2_select_all_ops
test_buck2_select_ops_in_list
test_buck2_select_ops_from_yaml

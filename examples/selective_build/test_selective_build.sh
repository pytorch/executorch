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

    echo "Running selective build test"
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=all -- --model_path=./mv3.pte

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_buck2_select_ops_in_list() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.export.export_example --model_name="add_mul"

    echo "Running selective build test"
    # set max_kernel_num=16: 13 primops, add, mul
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.max_kernel_num=16 \
        --config=executorch.select_ops=list -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_from_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.custom_ops.custom_ops_1

    echo "Running selective build test"
    buck2 run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=yaml -- --model_path=./custom_ops_1.pte

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

test_cmake_select_all_ops() {
    echo "Exporting MobilenetV3"
    ${PYTHON_EXECUTABLE} -m examples.export.export_example --model_name="mv3"

    (rm -rf cmake-out \
        && mkdir cmake-out \
        && cd cmake-out \
        && cmake -DBUCK2=buck2 \
            -DBUILD_SELECTIVE_BUILD_TEST=ON \
            -DSELECT_ALL_OPS=ON \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

    echo "Build selective build test"
    cmake --build cmake-out -j9

    echo 'Running selective build test'
    cmake-out/examples/selective_build/selective_build_test --model_path="./mv3.pte"

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_cmake_select_ops_in_list() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.export.export_example --model_name="add_mul"

    # set MAX_KERNEL_NUM=16: 13 primops, add, mul
    (rm -rf cmake-out \
        && mkdir cmake-out \
        && cd cmake-out \
        && cmake -DBUCK2=buck2 \
            -DMAX_KERNEL_NUM=16 \
            -DBUILD_SELECTIVE_BUILD_TEST=ON \
            -DSELECT_OPS_LIST="aten::add.out,aten::mm.out" \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

    echo "Build selective build test"
    cmake --build cmake-out -j9

    echo 'Running selective build test'
    cmake-out/examples/selective_build/selective_build_test --model_path="./add_mul.pte"

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_cmake_select_ops_in_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.custom_ops.custom_ops_1

    (rm -rf cmake-out \
        && mkdir cmake-out \
        && cd cmake-out \
        && cmake -DBUCK2=buck2 \
            -DBUILD_SELECTIVE_BUILD_TEST=ON \
            -DSELECT_OPS_YAML=ON \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

    echo "Build selective build test"
    cmake --build cmake-out -j9

    echo 'Running selective build test'
    cmake-out/examples/selective_build/selective_build_test --model_path="./custom_ops_1.pte"

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ $1 == "cmake" ]];
then
    test_cmake_select_all_ops
    test_cmake_select_ops_in_list
    test_cmake_select_ops_in_yaml
elif [[ $1 == "buck2" ]];
then
    test_buck2_select_all_ops
    test_buck2_select_ops_in_list
    test_buck2_select_ops_from_yaml
fi

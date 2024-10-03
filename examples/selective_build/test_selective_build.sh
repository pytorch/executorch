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

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../.ci/scripts/utils.sh"


# BUCK2 examples; test internally in fbcode/xplat
# 1. `--config executorch.select_ops=all`: select all ops from the dependency
#       kernel libraries, register all of them into ExecuTorch runtime.
# 2. `--config executorch.select_ops=list`: Only select ops from `ops` kwarg
#       in `et_operator_library` macro.
# 3. `--config executorch.select_ops=yaml`: Only select from a yaml file from
#       `ops_schema_yaml_target` kwarg in `et_operator_library` macro
# 4. `--config executorch.select_ops=dict`: Only select ops from `ops_dict`
#       kwarg in `et_operator_library` macro. Add `dtype_selective_build = True`
#       to executorch_generated_lib to select dtypes specified in the dictionary.

# Other configs:
# - `--config executorch.max_kernel_num=N`: Only allocate memory for the
#       required number of operators. Users can retrieve N from `selected_operators.yaml`.

test_buck2_select_all_ops() {
    echo "Exporting MobilenetV3"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="mv3"

    echo "Running selective build test"
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=all -- --model_path=./mv3.pte

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_buck2_select_ops_in_list() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="add_mul"

    echo "Running selective build test"
    # set max_kernel_num=22: 19 primops, add, mul
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.max_kernel_num=22 \
        --config=executorch.select_ops=list \
        -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_in_dict() {
    echo "Exporting add_mul"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="add_mul"

    echo "Running selective build test"
    # select ops and their dtypes using the dictionary API.
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=dict \
        --config=executorch.dtype_selective_build_lib=//examples/selective_build:select_ops_in_dict_lib \
        -- --model_path=./add_mul.pte

    echo "Removing add_mul.pte"
    rm "./add_mul.pte"
}

test_buck2_select_ops_from_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.portable.custom_ops.custom_ops_1

    echo "Running selective build test"
    $BUCK run //examples/selective_build:selective_build_test \
        --config=executorch.select_ops=yaml -- --model_path=./custom_ops_1.pte

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

# CMake examples; test in OSS. Check the README for more information.
test_cmake_select_all_ops() {
    echo "Exporting MobilenetV3"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="mv3"

    local example_dir=examples/selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_BUILD_TYPE=Release \
            -DEXECUTORCH_SELECT_ALL_OPS=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./mv3.pte"

    echo "Removing mv3.pte"
    rm "./mv3.pte"
}

test_cmake_select_ops_in_list() {
    echo "Exporting MobilenetV2"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="mv2"

    local example_dir=examples/selective_build
    local build_dir=cmake-out/${example_dir}
    # set MAX_KERNEL_NUM=22: 19 primops, add, mul
    rm -rf ${build_dir}
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_BUILD_TYPE=Release \
            -DMAX_KERNEL_NUM=22 \
            -DEXECUTORCH_SELECT_OPS_LIST="aten::convolution.out,\
aten::_native_batch_norm_legit_no_training.out,aten::hardtanh.out,aten::add.out,\
aten::mean.out,aten::view_copy.out,aten::permute_copy.out,aten::addmm.out,\
aten,aten::clone.out" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./mv2.pte"

    echo "Removing mv2.pte"
    rm "./mv2.pte"
}

test_cmake_select_ops_in_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.portable.custom_ops.custom_ops_1
    local example_dir=examples/selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_BUILD_TYPE=Release \
            -DEXECUTORCH_SELECT_OPS_YAML=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./custom_ops_1.pte"

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ $1 == "cmake" ]];
then
    cmake_install_executorch_lib
    test_cmake_select_all_ops
    test_cmake_select_ops_in_list
    test_cmake_select_ops_in_yaml
elif [[ $1 == "buck2" ]];
then
    test_buck2_select_all_ops
    test_buck2_select_ops_in_list
    test_buck2_select_ops_in_dict
    test_buck2_select_ops_from_yaml
fi

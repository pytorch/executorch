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

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"

test_cmake_custom_op_1() {
  local model_name='custom_ops_1'
  echo "Exporting ${model_name}.pte"
  ${PYTHON_EXECUTABLE} -m "examples.portable.custom_ops.${model_name}"
  # should save file custom_ops_1.pte
  local example_dir=examples/portable/custom_ops
  local build_dir=cmake-out/${example_dir}
  rm -rf ${build_dir}
  retry cmake \
        -DREGISTER_EXAMPLE_CUSTOM_OP=1 \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

  echo "Building ${example_dir}"
  cmake --build ${build_dir} -j9 --config Release

  echo 'Running custom_ops_executor_runner'
  ${build_dir}/custom_ops_executor_runner --model_path="./${model_name}.pte"
}

get_shared_lib_ext() {
  UNAME=$(uname)
  if [[ $UNAME == "Darwin" ]];
  then
    EXT=".dylib"
  elif [[ $UNAME == "Linux" ]];
  then
    EXT=".so"
  else
    echo "Unsupported platform $UNAME"
    exit 1
  fi
  echo $EXT
}

test_cmake_custom_op_2() {
  local model_name='custom_ops_2'
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="$PWD/cmake-out/lib/cmake/ExecuTorch;${SITE_PACKAGES}/torch"

  local example_dir=examples/portable/custom_ops
  local build_dir=cmake-out/${example_dir}
  rm -rf ${build_dir}
  retry cmake \
        -DREGISTER_EXAMPLE_CUSTOM_OP=2 \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

  echo "Building ${example_dir}"
  cmake --build ${build_dir} -j9 --config Release

  EXT=$(get_shared_lib_ext)
  echo "Exporting ${model_name}.pte"
  ${PYTHON_EXECUTABLE} -m "examples.portable.custom_ops.${model_name}" --so_library="cmake-out/examples/portable/custom_ops/libcustom_ops_aot_lib$EXT"
  # should save file custom_ops_2.pte

  echo 'Running custom_ops_executor_runner'
  ${build_dir}/custom_ops_executor_runner --model_path="./${model_name}.pte"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib
test_cmake_custom_op_1
test_cmake_custom_op_2

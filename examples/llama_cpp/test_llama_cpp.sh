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

cmake_install_executorch_lib() {
    echo "Installing libexecutorch.a and libportable_kernels.a"
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DCMAKE_BUILD_TYPE=Debug \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -Bcmake-out .
    cmake --build cmake-out -j9 --target install
}

cmake_install_llama_cpp() {
    echo "Installing llama.cpp"
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DCMAKE_BUILD_TYPE=Debug \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -Bcmake-out/examples/third-party/llama_cpp examples/third-party/llama_cpp
    cmake --build cmake-out/examples/third-party/llama_cpp -j9 --target install
}

cmake_run_llama_cpp_test() {
    # build and run llama_cpp_test
    retry cmake -DBUCK2="$BUCK" \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DCMAKE_BUILD_TYPE=Debug \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -Bcmake-out/examples/llama_cpp examples/llama_cpp
    cmake --build cmake-out/examples/llama_cpp -j9

    # Export model
    echo "Exporting llama2"
    $PYTHON_EXECUTABLE -m examples.llama_cpp.export

    # Run model
    cmake-out/examples/llama_cpp/llama_cpp_test --model_path="./llama2_fused.pte"
}

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib
cmake_install_llama_cpp
cmake_run_llama_cpp_test

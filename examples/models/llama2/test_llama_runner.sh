#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test Llama runner in examples/models/llama2/main.cpp
# 1. Export a llama-like model
# 2. Build llama runner binary
# 3. Run model with the llama runner binary with prompt
set -e
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out .
    cmake --build cmake-out -j9 --target install --config Release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    dir="examples/models/llama2"
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config Release

}

if [[ $1 == "cmake" ]];
then
    cmake_install_executorch_libraries
    cmake_build_llama_runner
    # TODO(larryliu0820): export a model and verify the result
fi

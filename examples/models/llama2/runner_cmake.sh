#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# build the llama 2 runner using cmake

set -e

source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"
cmake_install_executorch_sdk_lib() {
  echo "Installing libexecutorch.a, libportable_kernels.a, libetdump.a, libbundled_program.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_SDK=ON \
          -DEXECUTORCH_BUILD_MPS=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

cmake_llama_runner() {
    echo "Calling cmake_llama_runner"

    local example_dir=examples/models/llama2
    local build_dir=cmake-out/${example_dir}
    CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"

    # build llama2 runner
    rm -rf ${build_dir}
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

      echo "Building ${example_dir}"
  cmake --build ${build_dir} --config Release

  echo 'Running llama 2 runner'
  ${build_dir}/main
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi


cmake_install_executorch_sdk_lib
cmake_llama_runner

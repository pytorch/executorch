#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of building sdk_example_runner and use it to run
# an actual model.


set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../.ci/scripts/utils.sh"

cmake_install_executorch_sdk_lib() {
  echo "Installing libexecutorch.a, libportable_kernels.a, libetdump.a, libbundled_program.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_SDK=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

test_cmake_sdk_example_runner() {
  echo "Exporting MobilenetV2"
  ${PYTHON_EXECUTABLE} -m examples.sdk.scripts.export_bundled_program --model_name="mv2"
  local example_dir=examples/sdk
  local build_dir=cmake-out/${example_dir}
  CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
  rm -rf ${build_dir}
  retry cmake \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

  echo "Building ${example_dir}"
  cmake --build ${build_dir} -j9 --config Release

  echo 'Running sdk_example_runner'
  ${build_dir}/sdk_example_runner --bundled_program_path="./mv2_bundled.bpte"
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
test_cmake_sdk_example_runner

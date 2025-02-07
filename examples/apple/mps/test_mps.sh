#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of mps runner.

set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"
cmake_install_executorch_devtools_lib() {
  echo "Installing libexecutorch.a, libportable_kernels.a, libetdump.a, libbundled_program.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_DEVTOOLS=ON \
          -DEXECUTORCH_BUILD_MPS=ON \
          -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

test_cmake_mps() {
    echo "Exporting MobilenetV2"
    ${PYTHON_EXECUTABLE} -m examples.apple.mps.scripts.mps_example --model_name="mv2" --bundled

    local example_dir=examples/apple/mps
    local build_dir=cmake-out/${example_dir}
    CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
    # build mps_executor_runner
    rm -rf ${build_dir}
    retry cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

      echo "Building ${example_dir}"
  cmake --build ${build_dir} -j9 --config Release

  echo 'Running mps_executor_runner'
  ${build_dir}/mps_executor_runner --bundled_program=True --model_path="./mv2_mps_bundled.pte"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi


cmake_install_executorch_devtools_lib
test_cmake_mps

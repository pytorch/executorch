#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of qualcomm runner.

set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"
cmake_install_executorch_qnn_lib() {
  echo "Installing libexecutorch.a, libqnn_executorch_backend.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_QNN=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

test_cmake_qualcomm() {
    echo "Exporting MobilenetV2"
    ${PYTHON_EXECUTABLE} -m examples.qualcomm.scripts.export_example --model_name mv2

    local example_dir=examples/qualcomm
    local build_dir=cmake-out/${example_dir}
    CMAKE_PREFIX_PATH="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
    # build qnn_executor_runner
    rm -rf ${build_dir}
    retry cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI='arm64-v8a' \
        -DANDROID_NATIVE_API_LEVEL=23 \
        -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -B${build_dir} \
        ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release
    # Need to run on device
#   ${build_dir}/qnn_executor_runner --model_path="./mv2_qnn.pte"
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ -z $BUCK ]];
then
  BUCK=buck2
fi


cmake_install_executorch_qnn_lib
test_cmake_qualcomm

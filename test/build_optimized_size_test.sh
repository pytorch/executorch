#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Unlike build_size_test.sh, this script:
# - does not attempt to disable exceptions and RTTI
# - as a consequence, is able to build optimized kernels
# - uses MinSizeRel builds
# - is not currently intended to run in CI
# - sets -g to make it easier to use tools like bloaty to investigate size

set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../.ci/scripts/utils.sh"

cmake_install_executorch_lib() {
  echo "Installing libexecutorch.a"
  clean_executorch_install_folders
  update_tokenizers_git_submodule
  CXXFLAGS="-g" retry cmake \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=MinSizeRel \
          -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
          -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
          -DEXECUTORCH_OPTIMIZE_SIZE=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config MinSizeRel
}

test_cmake_size_test() {
    CXXFLAGS="-g" retry cmake -DCMAKE_BUILD_TYPE=MinSizeRel -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DCMAKE_INSTALL_PREFIX=cmake-out -Bcmake-out/test test

    echo "Build size test"
    cmake --build cmake-out/test -j9 --config MinSizeRel

    echo 'ExecuTorch with no ops binary size, unstripped:'
    ls -al cmake-out/test/size_test
    size cmake-out/test/size_test

    echo 'ExecuTorch with portable ops binary size, unstripped:'
    ls -al cmake-out/test/size_test_all_ops
    size cmake-out/test/size_test_all_ops

    echo 'ExecuTorch with optimized ops binary size, unstripped:'
    ls -al cmake-out/test/size_test_all_optimized_ops
    size cmake-out/test/size_test_all_optimized_ops
}

if [[ -z $PYTHON_EXECUTABLE ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib
test_cmake_size_test

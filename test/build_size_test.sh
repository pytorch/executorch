#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build size_test and show the size of it
set -ex

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../.ci/scripts/utils.sh"

EXTRA_BUILD_ARGS="${@:-}"
# TODO(#8357): Remove -Wno-int-in-bool-context
# TODO: Replace -ET_HAVE_PREAD=0 with a CMake option.
#  FileDataLoader used in the size_test breaks baremetal builds with pread when missing.
COMMON_CXXFLAGS="-fno-exceptions -fno-rtti -Wall -Werror -Wno-int-in-bool-context -DET_HAVE_PREAD=0 -Wno-stringop-overread"

cmake_install_executorch_lib() {
  echo "Installing libexecutorch.a"
  clean_executorch_install_folders
  update_tokenizers_git_submodule
  local EXTRA_BUILD_ARGS="${@}"

  CXXFLAGS="$COMMON_CXXFLAGS" retry cmake \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
          -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
          -DEXECUTORCH_OPTIMIZE_SIZE=ON \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          ${EXTRA_BUILD_ARGS} \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

test_cmake_size_test() {
    CXXFLAGS="$COMMON_CXXFLAGS" retry cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        ${EXTRA_BUILD_ARGS} \
        -Bcmake-out/test test

    echo "Build size test"
    cmake --build cmake-out/test -j9 --config Release

    echo 'ExecuTorch with no ops binary size, unstripped:'
    ls -al cmake-out/test/size_test
    size cmake-out/test/size_test

    echo 'ExecuTorch with portable ops binary size, unstripped:'
    ls -al cmake-out/test/size_test_all_ops
    size cmake-out/test/size_test_all_ops
}

if [[ -z $PYTHON_EXECUTABLE ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib ${EXTRA_BUILD_ARGS}
test_cmake_size_test ${EXTRA_BUILD_ARGS}

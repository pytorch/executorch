#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build size_test and show the size of it
set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../.ci/scripts/utils.sh"

cmake_install_executorch_lib() {
  echo "Installing libexecutorch.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK2" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
          -Bcmake-out .
  cmake --build cmake-out -j9 --target install --config Release
}

test_cmake_size_test() {
    retry cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=cmake-out -Bcmake-out/test test

    echo "Build selective build test"
    cmake --build cmake-out/test -j9 --config Release

    echo 'Size of the binary:'
    ls -al cmake-out/test/size_test
}

if [[ -z $BUCK2 ]]; then
  BUCK2=buck2
fi

if [[ -z $PYTHON_EXECUTABLE ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_executorch_lib
test_cmake_size_test

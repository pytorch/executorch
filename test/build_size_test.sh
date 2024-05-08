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

# Set compile flags for Clang and GCC.
# -Wno-gnu allows us to use gnu statement-expressions.
# -Werror -Wc++17* ensure we do not use features from C++17.
CXX_FLAGS="-Wno-gnu"
compiler=$(cc --version)
if [[ $compiler == *"clang"* ]]; then
  CXX_FLAGS="$CXX_FLAGS -Werror -Wc++17-extensions -Wc++14-extensions"
elif [[ $compiler == *"cc"* ]]; then
  CXX_FLAGS="$CXX_FLAGS -Werror -Wc++17-compat -Wc++14-compat"
else
  echo "Unknown compiler: $compiler"
  exit 1
fi
echo "Using compiler $compiler with flags $CXX_FLAGS"

cmake_install_executorch_lib() {
  echo "Installing libexecutorch.a"
  rm -rf cmake-out

  retry cmake -DBUCK2="$BUCK2" \
          -DCMAKE_CXX_STANDARD=11 \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
          -DCMAKE_INSTALL_PREFIX=cmake-out \
          -DCMAKE_BUILD_TYPE=Release \
          -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
          -DOPTIMIZE_SIZE=ON \
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

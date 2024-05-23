#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script helps build and run C++ tests with CMakeLists.txt.
# It builds and installs the root ExecuTorch package, and then sub-directories.
#
# If no arg is given, it probes all sub-directories containing
# test/CMakeLists.txt. It builds and runs these tests.
# If an arg is given, like `runtime/core/test/`, it runs that directory only.

set -ex

build_executorch() {
  cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -Bcmake-out
  cmake --build cmake-out -j9 --target install
}

build_gtest() {
  git clone https://github.com/google/googletest.git -b v1.14.0 cmake-out/googletest
  mkdir cmake-out/googletest/build
  pushd cmake-out/googletest/build
  cmake .. -DCMAKE_INSTALL_PREFIX=.
  make -j4
  make install
  popd
}

build_and_run_test() {
  local test_dir=$1
  cmake "${test_dir}" \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DGTest_DIR="$(pwd)/googletest/cmake-out/build/lib/cmake/GTest" --debug-find \
    -Bcmake-out/"${test_dir}"
  cmake --build cmake-out/"${test_dir}" -j9 || true
  cat cmake-out/"${test_dir}"/CMakeFiles/*.dir/link.txt
  for t in cmake-out/"${test_dir}"/*test; do ./"$t"; done
}

probe_tests() {
  # This function finds the set of directories that contain C++ tests
  # CMakeLists.txt rules, that are buildable using build_and_run_test
  dirs=(
    backends
    examples
    extension
    kernels
    runtime
    schema
    sdk
    test
  )

  find "${dirs[@]}" \
      \( -type f -wholename '*/test/CMakeLists.txt' -exec dirname {} \; \) -o \
      \( -type d -path '*/third-party/*' -prune \) \
      | sort -u
}

build_executorch
build_gtest

if [ -z "$1" ]; then
  echo "Running all directories:"
  probe_tests

  for test_dir in $(probe_tests); do
    build_and_run_test "${test_dir}"
  done
else
  build_and_run_test "$1"
fi

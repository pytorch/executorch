#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

build_executorch() {
  cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_BUILD_GTESTS=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -Bcmake-out
  cmake --build cmake-out -j9 --target install
}

build_and_run_test() {
  local test_dir=$1
  cmake "${test_dir}" -Bcmake-out/"${test_dir}" -DCMAKE_INSTALL_PREFIX=cmake-out
  cmake --build cmake-out/"${test_dir}" -j9
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

echo "Found test directories:"
echo "$(probe_tests)"

for test_dir in $(probe_tests); do
  build_and_run_test $test_dir
done

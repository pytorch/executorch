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

if [[ $(uname) == "Darwin" ]]; then
  export LLVM_PROFDATA="${LLVM_PROFDATA:-xcrun llvm-profdata}"
  export LLVM_COV="${LLVM_COV:-xcrun llvm-cov}"
elif [[ $(uname) == "Linux" ]]; then
  export LLVM_PROFDATA="${LLVM_PROFDATA:-llvm-profdata}"
  export LLVM_COV="${LLVM_COV:-llvm-cov}"
fi

build_executorch() {
  BUILD_VULKAN="OFF"
  if [ -x "$(command -v glslc)" ]; then
    BUILD_VULKAN="ON"
  fi
  cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_VULKAN=$BUILD_VULKAN \
    -Bcmake-out
  cmake --build cmake-out -j9 --target install
}

build_gtest() {
  mkdir -p third-party/googletest/build
  pushd third-party/googletest/build
  cmake .. -DCMAKE_INSTALL_PREFIX=.
  make -j4
  make install
  popd
}

build_and_run_test() {
  local test_dir=$1
  cmake "${test_dir}" \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_USE_CPP_CODE_COVERAGE=ON \
    -DCMAKE_PREFIX_PATH="$(pwd)/third-party/googletest/build" \
    -Bcmake-out/"${test_dir}"
  cmake --build cmake-out/"${test_dir}" -j9

  export RESOURCES_PATH=extension/module/test/resources

  for t in cmake-out/"${test_dir}"/*test; do
    if [ -e "$t" ]; then
      LLVM_PROFILE_FILE="cmake-out/$(basename $t).profraw" ./"$t";
      TEST_BINARY_LIST="${TEST_BINARY_LIST} -object $t"
    fi
  done
}

report_coverage() {
  "${LLVM_PROFDATA}" merge -sparse cmake-out/*.profraw -o cmake-out/merged.profdata
  "${LLVM_COV}" report -instr-profile=cmake-out/merged.profdata $TEST_BINARY_LIST
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

report_coverage || true

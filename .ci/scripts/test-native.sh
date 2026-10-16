#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

component="${1:?Usage: test-native.sh <runtime|cpu|ggml|vulkan> [cmake options]}"
shift
build_dir="cmake-out/native-${component}"
build_jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
cmake_args=()
targets=()

case "${component}" in
  runtime)
    cmake_args+=(-DEXECUTORCH_BUILD_NATIVE=ON)
    targets+=(native_runtime_test)
    test_regex='^native_runtime_test$'
    ;;
  cpu)
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_CPU=ON
      -DEXECUTORCH_BUILD_XNNPACK=ON
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
      -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
    )
    targets+=(native_cpu_engine_test run_classification_cpu)
    test_regex='^native_cpu_engine_test$'
    ;;
  ggml)
    cmake_args+=(-DEXECUTORCH_BUILD_NATIVE_GGML=ON -DGGML_NATIVE=OFF)
    targets+=(native_ggml_engine_test)
    test_regex='^native_ggml_engine_test$'
    ;;
  vulkan)
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON
      -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
      -DEXECUTORCH_VULKAN_SHADER_COMPILE_NTHREADS="${build_jobs}"
    )
    targets+=(native_vulkan_engine run_classification native_vulkan_passes_test native_runner_utils_test)
    test_regex='^native_(vulkan_passes|runner_utils)_test$'
    ;;
  *)
    echo "Unknown native component: ${component}" >&2
    exit 1
    ;;
esac

cmake -S . -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DEXECUTORCH_BUILD_TESTS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  "${cmake_args[@]}" "$@"
cmake --build "${build_dir}" \
  --parallel "${build_jobs}" \
  --target "${targets[@]}"
ctest --test-dir "${build_dir}/backends/native/test" \
  -R "${test_regex}" --output-on-failure --no-tests=error --timeout 120

#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euxo pipefail

component="${1:?Usage: test-native.sh <runtime|ggml|vulkan> [cmake options]}"
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
  ggml)
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_GGML=ON
      -DEXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON
      -DGGML_NATIVE=OFF
    )
    targets+=(run_hf_llm_ggml native_ggml_engine_test native_llm_runner_utils_test)
    test_regex='^native_(ggml_engine|llm_runner_utils)_test$'
    ;;
  vulkan)
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_VULKAN=ON
      -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
      -DEXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON
      -DEXECUTORCH_VULKAN_SHADER_COMPILE_NTHREADS="${build_jobs}"
    )
    targets+=(native_vulkan_engine run_classification run_hf_llm_vulkan native_vulkan_passes_test native_runner_utils_test)
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
ctest --test-dir "${build_dir}/backends/native" \
  -R "${test_regex}" --output-on-failure --no-tests=error --timeout 120

if [[ "${component}" == "ggml" ]]; then
  PYTHONPATH="${PWD}/src:${PYTHONPATH:-}" \
    FLATC_EXECUTABLE="${FLATC_EXECUTABLE:-${PWD}/${build_dir}/third-party/flatc_ep/bin/flatc}" \
    python .ci/scripts/test_native_llm_runner.py \
      --runner "${build_dir}/backends/native/examples/hf_llm/runtime/run_hf_llm_ggml"
fi

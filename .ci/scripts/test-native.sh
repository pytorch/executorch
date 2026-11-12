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
    targets+=(native_runtime_test)
    test_regex='^native_runtime_test$'
    ;;
  cpu)
    # The CPU plan test covers ETDump tracing, including operator shapes.
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_CPU=ON
      -DEXECUTORCH_BUILD_XNNPACK=ON
      -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
      -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
      -DEXECUTORCH_BUILD_DEVTOOLS=ON
      -DEXECUTORCH_ENABLE_EVENT_TRACER=ON
      -DEXECUTORCH_CPU_XNNPACK_OPERATOR_SHAPES=ON
    )
    targets+=(native_cpu_engine_test cpu_plan_test run_classification_cpu)
    test_regex='^(native_cpu_engine|cpu_plan)_test$'
    ;;
  ggml)
    cmake_args+=(
      -DEXECUTORCH_BUILD_NATIVE_GGML=ON
      -DEXECUTORCH_BUILD_NATIVE_AUDIO_RUNNER=ON
      -DEXECUTORCH_BUILD_NATIVE_LLM_RUNNER=ON
      -DEXECUTORCH_BUILD_NATIVE_VISION_RUNNER=ON
      -DGGML_NATIVE=OFF
    )
    # TODO: run the other sample tests once they pass.
    targets+=(run_hf_llm_ggml run_whisper_ggml_cpu native_ggml_engine_test native_llm_runner_utils_test run_classification_ggml_cpu native_output_check_test native_ggml_sample_test)
    test_regex='^native_(ggml_engine|llm_runner_utils|output_check)_test$|^native_ggml_sample_test_act_quant$'
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
  -DEXECUTORCH_BUILD_NATIVE=ON \
  "${cmake_args[@]}" "$@"
cmake --build "${build_dir}" \
  --parallel "${build_jobs}" \
  --target "${targets[@]}"
ctest --test-dir "${build_dir}" \
  -R "${test_regex}" --output-on-failure --no-tests=error --timeout 120

if [[ "${component}" == "ggml" ]]; then
  export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
  export FLATC_EXECUTABLE="${FLATC_EXECUTABLE:-${PWD}/${build_dir}/third-party/flatc_ep/bin/flatc}"
  python .ci/scripts/test_native_llm_runner.py \
    --runner "${build_dir}/backends/native/examples/hf_llm/runtime/run_hf_llm_ggml"
  NATIVE_WHISPER_RUNNER="${PWD}/${build_dir}/backends/native/examples/audio/run_whisper_ggml_cpu" \
    python -m pytest \
    backends/native/test/test_hf_whisper.py::WhisperNativeExportTest::test_ggml_runner_matches_eager
fi

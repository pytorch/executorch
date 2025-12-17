#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# Initialize variables
RUN_BUILD=false
RUN_CLEAN=false
RUN_CLEAN_TESTS=false
RUN_RECOMPILE=false
RUN_TESTS=false
TEST_BINARY=""
ATEN_OP=""

# Parse arguments
SKIP_NEXT=false
if [[ $# -eq 0 ]]; then
  # No arguments provided - run default test
  TEST_BINARY="vulkan_op_correctness_tests"
  RUN_TESTS=true
else
  for i in $(seq 1 $#); do
    if [[ "$SKIP_NEXT" == true ]]; then
      SKIP_NEXT=false
      continue
    fi

    arg="${!i}"
    case $arg in
      --build|-b)
        RUN_BUILD=true
        ;;
      --clean|-c)
        RUN_CLEAN=true
        RUN_BUILD=true
        ;;
      --clean_tests|-ct)
        RUN_CLEAN_TESTS=true
        ;;
      --recompile|-rc)
        RUN_RECOMPILE=true
        ;;
      --test|-t)
        RUN_TESTS=true
        ;;
      --aten)
        next_i=$((i + 1))
        if [[ $next_i -le $# ]]; then
          ATEN_OP="${!next_i}"
          TEST_BINARY="vulkan_op_correctness_tests"
          RUN_TESTS=true
          SKIP_NEXT=true
        else
          echo "Error: --aten requires an operator name"
          exit 1
        fi
        ;;
      --*|-*)
        echo "Unknown argument: $arg"
        exit 1
        ;;
      *)
        if [[ -z "$TEST_BINARY" ]]; then
          TEST_BINARY="$arg"
          RUN_TESTS=true
        else
          echo "Multiple test binaries provided: $TEST_BINARY and $arg"
          exit 1
        fi
        ;;
    esac
  done
fi

# Determine execution mode based on parsed arguments
if [[ "$RUN_BUILD" == true ]] && [[ -z "$TEST_BINARY" ]] && [[ "$RUN_TESTS" == false ]]; then
  # Build-only mode
  echo "Build-only mode"
elif [[ "$RUN_BUILD" == true ]] && [[ -n "$TEST_BINARY" ]]; then
  # Build and test mode
  echo "Build and test mode for: $TEST_BINARY"
elif [[ "$RUN_BUILD" == false ]] && [[ -n "$TEST_BINARY" ]]; then
  # Test-only mode
  echo "Test-only mode for: $TEST_BINARY"
elif [[ "$RUN_TESTS" == true ]] && [[ -z "$TEST_BINARY" ]]; then
  # Run all available tests
  echo "Running all available operator tests"
elif [[ $# -eq 0 ]]; then
  # No arguments provided - run default test
  TEST_BINARY="vulkan_op_correctness_tests"
  RUN_TESTS=true
  echo "No arguments provided, running default test: $TEST_BINARY"
else
  echo "Invalid argument combination. Usage:"
  echo "  $0                                                                              # Run default vulkan_op_correctness_tests"
  echo "  $0 --build|-b [--clean|-c] [--clean_tests|-ct] [--recompile|-rc]                # Build-only mode"
  echo "  $0 [test_binary_name] [--build|-b] [--clean|-c] [--clean_tests|-ct] [--recompile|-rc]  # Test mode or build+test mode"
  echo "  $0 --test|-t [--build|-b] [--clean|-c] [--clean_tests|-ct] [--recompile|-rc]    # Run all tests mode"
  echo "  $0 --aten <operator_name> [--build|-b] [--clean|-c] [--clean_tests|-ct] [--recompile|-rc]  # Run specific ATen operator test"
  echo "  $0 --clean_tests|-ct                                                            # Clean and rebuild only operator tests"
  echo ""
  echo "Available test binaries:"
  echo "  - vulkan_op_correctness_tests"
  echo "  - vulkan_op_benchmarks"
  echo "  - compute_graph_op_tests"
  echo "  - sdpa_test"
  exit 1
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

CMAKE_OUTPUT_DIR=cmake-out

clean_build_directory() {
  echo "Cleaning build directory: ${CMAKE_OUTPUT_DIR}"
  rm -rf ${CMAKE_OUTPUT_DIR}
}

clean_test_directory() {
  echo "Cleaning test build directory: ${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests"
  rm -rf ${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests
}

build_core_libraries() {
  cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM_AOT=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_TESTS=ON \
    -Bcmake-out && \
  cmake --build cmake-out -j64 --target install
}

build_operator_tests() {
  echo "Building Vulkan operator tests..."

  # Prepare CMAKE arguments
  CMAKE_ARGS=(
    "backends/vulkan/test/op_tests"
    "-DCMAKE_INSTALL_PREFIX=cmake-out"
    "-DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE"
    "-DCMAKE_CXX_STANDARD=17"
  )

  # Check if TORCH_OPS_YAML_PATH is set
  if [[ -n "${TORCH_OPS_YAML_PATH:-}" ]]; then
    # Verify that TORCH_OPS_YAML_PATH exists
    if [[ ! -d "$TORCH_OPS_YAML_PATH" ]]; then
      echo "Error: TORCH_OPS_YAML_PATH directory does not exist: $TORCH_OPS_YAML_PATH"
      echo "Please set TORCH_OPS_YAML_PATH to a valid PyTorch native operations directory"
      echo "Example: export TORCH_OPS_YAML_PATH=/path/to/pytorch/aten/src/ATen/native"
      exit 1
    fi

    # Verify required YAML files exist
    if [[ ! -f "$TORCH_OPS_YAML_PATH/native_functions.yaml" ]]; then
      echo "Error: Required file not found: $TORCH_OPS_YAML_PATH/native_functions.yaml"
      exit 1
    fi

    if [[ ! -f "$TORCH_OPS_YAML_PATH/tags.yaml" ]]; then
      echo "Error: Required file not found: $TORCH_OPS_YAML_PATH/tags.yaml"
      exit 1
    fi

    echo "Using TORCH_OPS_YAML_PATH: $TORCH_OPS_YAML_PATH"
    CMAKE_ARGS+=("-DTORCH_OPS_YAML_PATH=$TORCH_OPS_YAML_PATH")
  else
    echo "WARNING: TORCH_OPS_YAML_PATH is not set. Building without PyTorch operator definitions."
    echo "Some functionality may be limited. To enable full functionality, set TORCH_OPS_YAML_PATH to point to PyTorch's native operations directory."
    echo "Example: export TORCH_OPS_YAML_PATH=/path/to/pytorch/aten/src/ATen/native"
  fi

  # Build operator tests
  cmake "${CMAKE_ARGS[@]}" \
    -Bcmake-out/backends/vulkan/test/op_tests && \
  cmake --build cmake-out/backends/vulkan/test/op_tests -j16
}

recompile() {
  echo "Recompiling..."
  cmake --build cmake-out -j64 --target install
  cmake --build cmake-out/backends/vulkan/test/op_tests -j16
}

run_operator_test() {
  local test_name="$1"
  local test_binary_path=""

  case "$test_name" in
    "aten")
      test_binary_path="${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests/vulkan_op_correctness_tests"
      ;;
    *)
      # Try to find the binary directly
      test_binary_path="${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests/${test_name}"
      ;;
  esac

  if [[ -f "$test_binary_path" ]]; then
    echo "Running test binary: $test_binary_path"

    # Add gtest filter if ATEN_OP is specified
    if [[ -n "$ATEN_OP" ]]; then
      echo "Filtering tests for ATen operator: $ATEN_OP"
      "$test_binary_path" --gtest_filter="*${ATEN_OP}*"
    else
      "$test_binary_path"
    fi
  else
    echo "Error: Test binary not found at $test_binary_path"
    echo "Available binaries in ${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests/:"
    ls -la "${CMAKE_OUTPUT_DIR}/backends/vulkan/test/op_tests/" 2>/dev/null || echo "Directory not found"
    exit 1
  fi
}

# Main execution
if [[ "${RUN_CLEAN_TESTS}" == true ]]; then
  clean_test_directory
  build_operator_tests
fi

if [[ "${RUN_BUILD}" == true ]]; then
  if [[ "${RUN_CLEAN}" == true ]]; then
    clean_build_directory
  fi
  build_core_libraries
  build_operator_tests
fi

if [[ "${RUN_RECOMPILE}" == true ]]; then
  recompile
fi

if [[ "${RUN_TESTS}" == true ]]; then
  run_operator_test "$TEST_BINARY"

  # Check if tests completed successfully
  if [[ $? -eq 0 ]]; then
    echo "Vulkan operator tests completed successfully!"
  else
    echo "Some Vulkan operator tests failed!"
    exit 1
  fi
fi

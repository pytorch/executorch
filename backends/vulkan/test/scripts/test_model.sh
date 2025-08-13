#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# Initialize variables
RUN_BUILD=false
RUN_CORRECTNESS_TEST=false
RUN_CLEAN=false
RUN_RECOMPILE=false
MODEL_NAME=""
OUTPUT_DIRECTORY="."

# Parse arguments
SKIP_NEXT=false
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
      ;;
    --recompile|-rc)
      RUN_RECOMPILE=true
      ;;
    --output_directory|-o)
      next_i=$((i + 1))
      if [[ $next_i -le $# ]]; then
        OUTPUT_DIRECTORY="${!next_i}"
        SKIP_NEXT=true
      else
        echo "Error: --output_directory|-o requires a value"
        exit 1
      fi
      ;;
    --*|-*)
      echo "Unknown argument: $arg"
      exit 1
      ;;
    *)
      if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME="$arg"
      else
        echo "Multiple model names provided: $MODEL_NAME and $arg"
        exit 1
      fi
      ;;
  esac
done

# Determine execution mode based on parsed arguments
if [[ "$RUN_BUILD" == true ]] && [[ -z "$MODEL_NAME" ]]; then
  # Build-only mode
  RUN_CORRECTNESS_TEST=false
elif [[ "$RUN_BUILD" == true ]] && [[ -n "$MODEL_NAME" ]]; then
  # Build and test mode
  RUN_CORRECTNESS_TEST=true
elif [[ "$RUN_BUILD" == false ]] && [[ -n "$MODEL_NAME" ]]; then
  # Test-only mode
  RUN_CORRECTNESS_TEST=true
else
  echo "Invalid argument combination. Usage:"
  echo "  $0 --build|-b [--clean|-c] [--recompile|-rc] [-o|--output_directory DIR]                    # Build-only mode"
  echo "  $0 model_name [--build|-b] [--clean|-c] [--recompile|-rc] [-o|--output_directory DIR]       # Test mode or build+test mode"
  exit 1
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi
which "${PYTHON_EXECUTABLE}"

CMAKE_OUTPUT_DIR=cmake-out

# Only set EXPORTED_MODEL if running correctness test
if [[ "${RUN_CORRECTNESS_TEST}" == true ]]; then
  EXPORTED_MODEL=${MODEL_NAME}_vulkan
fi


clean_build_directory() {
  echo "Cleaning build directory: ${CMAKE_OUTPUT_DIR}"
  rm -rf ${CMAKE_OUTPUT_DIR}
}

recompile() {
  cmake --build cmake-out -j64 --target install
}

build_core_libraries_and_devtools() {
  echo "Building core libraries and devtools with comprehensive Vulkan support..."

  # Build core libraries with all required components
  cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM_AOT=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -Bcmake-out && \
  cmake --build cmake-out -j64 --target install

  # Build devtools example runner
  cmake examples/devtools \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -Bcmake-out/examples/devtools && \
  cmake --build cmake-out/examples/devtools -j16 --config Release
}

run_example_runner() {
  ./${CMAKE_OUTPUT_DIR}/examples/devtools/example_runner -bundled_program_path "${OUTPUT_DIRECTORY}/${EXPORTED_MODEL}.bpte" -output_verification
}

test_bundled_model_with_vulkan() {
  # Export model as bundled program with Vulkan backend
  "${PYTHON_EXECUTABLE}" -m examples.vulkan.export --model_name="${MODEL_NAME}" --output_dir="${OUTPUT_DIRECTORY}" --bundled

  # Update exported model name for bundled program
  EXPORTED_MODEL="${MODEL_NAME}_vulkan"

  # Verify the exported bundled model exists
  if [[ ! -f "${OUTPUT_DIRECTORY}/${EXPORTED_MODEL}.bpte" ]]; then
    echo "Error: Failed to export bundled model ${MODEL_NAME} with Vulkan backend"
    exit 1
  fi

  # Note: Running bundled programs may require different executor runner
  echo "Bundled program created successfully. Use appropriate bundled program runner to test."

  run_example_runner
}


# Main execution
if [[ "${RUN_BUILD}" == true ]]; then
  if [[ "${RUN_CLEAN}" == true ]]; then
    clean_build_directory
  fi
  build_core_libraries_and_devtools
fi

if [[ "${RUN_RECOMPILE}" == true ]]; then
  recompile
fi

if [[ "${RUN_CORRECTNESS_TEST}" == true ]]; then
  echo "Testing ${MODEL_NAME} with Vulkan backend..."
  # Always use bundled program testing
  test_bundled_model_with_vulkan

  # Check if test completed successfully
  if [[ $? -eq 0 ]]; then
    echo "Vulkan model test completed successfully!"
  else
    echo "Vulkan model test failed!"
    exit 1
  fi
fi

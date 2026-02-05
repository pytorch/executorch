#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to build and run Metal backend tests
# Usage:
#   ./run_metal_test.sh --build              # Build the Metal runtime
#   ./run_metal_test.sh --run <pte>          # Run inference with given model file
#   ./run_metal_test.sh --check-build        # Check if runtime is already built

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXECUTORCH_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BUILD_DIR="$EXECUTORCH_ROOT/cmake-out"
EXECUTOR_RUNNER="$BUILD_DIR/executor_runner"

# Function to check if Metal runtime is built
check_build() {
    if [[ -f "$EXECUTOR_RUNNER" ]]; then
        echo "true"
        return 0
    else
        echo "false"
        return 1
    fi
}

# Function to build the Metal runtime
build_runtime() {
    echo "Building Metal runtime..."

    # Check if we're on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        echo "Error: Metal backend is only supported on macOS"
        exit 1
    fi

    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # CMake configuration for Metal backend
    CMAKE_ARGS="-DEXECUTORCH_BUILD_METAL=ON \
                -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
                -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
                -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
                -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
                -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
                -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
                -DAOTI_METAL=ON \
                -DEXECUTORCH_LOG_LEVEL=Info \
                -DCMAKE_BUILD_TYPE=Release"

    echo "Running cmake..."
    cmake $CMAKE_ARGS "$EXECUTORCH_ROOT"

    echo "Building..."
    cmake --build . -j$(sysctl -n hw.ncpu)

    cd "$EXECUTORCH_ROOT"

    if [[ -f "$EXECUTOR_RUNNER" ]]; then
        echo "Build successful: $EXECUTOR_RUNNER"
    else
        echo "Error: Build failed - executor_runner not found"
        exit 1
    fi
}

# Function to run inference
run_inference() {
    local pte_path="$1"

    if [[ ! -f "$EXECUTOR_RUNNER" ]]; then
        echo "Error: executor_runner not found at $EXECUTOR_RUNNER"
        echo "Run '$0 --build' first to build the Metal runtime"
        exit 1
    fi

    if [[ ! -f "$pte_path" ]]; then
        echo "Error: PTE file not found: $pte_path"
        exit 1
    fi

    echo "Running inference..."
    echo "  PTE: $pte_path"

    "$EXECUTOR_RUNNER" --model_path "$pte_path"
}

# Parse command line arguments
case "$1" in
    --build)
        build_runtime
        ;;
    --run)
        if [[ -z "$2" ]]; then
            echo "Usage: $0 --run <pte_path>"
            exit 1
        fi
        run_inference "$2"
        ;;
    --check-build)
        check_build
        ;;
    *)
        echo "Metal Backend Test Runner"
        echo ""
        echo "Usage:"
        echo "  $0 --build              Build the Metal runtime"
        echo "  $0 --run <pte>          Run inference with given model file"
        echo "  $0 --check-build        Check if runtime is already built"
        exit 1
        ;;
esac

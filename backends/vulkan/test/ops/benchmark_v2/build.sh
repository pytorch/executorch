#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Builds ExecuTorch with the Vulkan backend, then etvk_ops_bench, with CMake.
# Run from the ExecuTorch root. Requires glslc on PATH, and ANDROID_NDK with
# --android.
#
# Usage: backends/vulkan/test/ops/benchmark_v2/build.sh [--android]
#
# The executable is written to
# <build dir>/backends/vulkan/test/ops/benchmark_v2/etvk_ops_bench, where the
# build directory is cmake-out, or cmake-android-out with --android.

set -eux

BUILD_DIR=cmake-out
PLATFORM_ARGS=()
if [[ "${1:-}" == "--android" ]]; then
  BUILD_DIR=cmake-android-out
  PLATFORM_ARGS=(
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
    -DANDROID_ABI=arm64-v8a
    -DANDROID_PLATFORM=android-28
  )
fi

GLSLC_PATH=$(command -v glslc)
JOBS=$(($(nproc 2>/dev/null || sysctl -n hw.ncpu) + 1))
BENCH_DIR=backends/vulkan/test/ops/benchmark_v2

cmake . \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DGLSLC_PATH="$GLSLC_PATH" \
  "${PLATFORM_ARGS[@]}" \
  -B"$BUILD_DIR"
cmake --build "$BUILD_DIR" -j"$JOBS" --target install

cmake "$BENCH_DIR" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGLSLC_PATH="$GLSLC_PATH" \
  "${PLATFORM_ARGS[@]}" \
  -B"$BUILD_DIR/$BENCH_DIR"
cmake --build "$BUILD_DIR/$BENCH_DIR" -j"$JOBS"

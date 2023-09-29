#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

EXECUTORCH_ROOT_PATH="$SCRIPT_DIR_PATH/../../../"
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/coreml"
IOS_TOOLCHAIN_PATH="$COREML_DIR_PATH/third-party/ios-cmake/ios.toolchain.cmake"
CMAKE_BUILD_DIR_PATH="$COREML_DIR_PATH/cmake-ios-out"
LIBRARIES_DIR_PATH="$COREML_DIR_PATH/runtime/libraries"
INCLUDE_DIR_PATH="$COREML_DIR_PATH/runtime/include"

echo "Executorch: Building executorchcoreml_runner "
echo "Executorch: Removing build directory $CMAKE_BUILD_DIR_PATH"
rm -rf "$CMAKE_BUILD_DIR_PATH"

# Build coremldelegate
echo "Executorch: Building coremldelegate"
cmake "$EXECUTORCH_ROOT_PATH"  -B"$CMAKE_BUILD_DIR_PATH" -DCMAKE_TOOLCHAIN_FILE="$IOS_TOOLCHAIN_PATH" -DPLATFORM=MAC_UNIVERSAL -DDEPLOYMENT_TARGET=13.0 -DEXECUTORCH_BUILD_COREML_DELGATE=ON -DEXECUTORCH_BUILD_XNNPACK=OFF
cmake --build "$CMAKE_BUILD_DIR_PATH"  -j9 -t coremldelegate

# Copy include headers
echo "Executorch: Copying headers"
mkdir "$INCLUDE_DIR_PATH"
cp -r "$CMAKE_BUILD_DIR_PATH/third-party/gflags/include/" "$INCLUDE_DIR_PATH"

# Copy required libraries
echo "Executorch: Copying libraries"
mkdir "$SCRIPT_PATH/../runtime/libraries"
cp -f "$CMAKE_BUILD_DIR_PATH/libexecutorch.a" "$LIBRARIES_DIR_PATH"
cp -f "$CMAKE_BUILD_DIR_PATH/third-party/gflags/libgflags_nothreads_debug.a" "$LIBRARIES_DIR_PATH"

# Build the runner
echo "Executorch: Building executorchcoreml_runner"
XCODE_WORKSPACE_DIR_PATH="$COREML_DIR_PATH/runtime/runner"
XCODE_BUILD_DIR_PATH="$COREML_DIR_PATH/xcode-runner-build"

xcodebuild build -workspace "$XCODE_WORKSPACE_DIR_PATH/executorchcoreml.xcworkspace" -scheme executorchcoreml_runner BUILD_DIR="$XCODE_BUILD_DIR_PATH"
mv -f "$XCODE_BUILD_DIR_PATH/DEBUG/executorchcoreml_runner" "$COREML_DIR_PATH"

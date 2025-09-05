#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


set -eux

MODE="Release"

usage() {
  echo "Builds Core ML executor runner"
  echo "Options:"
  echo "  --Debug              Use Debug build mode. Default: 'Release'"
  echo "Example:"
  echo "  $0 --Debug"
  exit 0
}

for arg in "$@"; do
  case $arg in
      -h|--help) usage ;;
      --Debug) MODE="Debug" ;;
      *)
  esac
done


EXECUTORCH_ROOT_PATH=$(git rev-parse --show-toplevel)
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"
EXAMPLES_COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/examples/apple/coreml"
IOS_TOOLCHAIN_PATH="$EXECUTORCH_ROOT_PATH/third-party/ios-cmake/ios.toolchain.cmake"
CMAKE_BUILD_DIR_PATH="$EXAMPLES_COREML_DIR_PATH/cmake-out"
LIBRARIES_DIR_PATH="$EXAMPLES_COREML_DIR_PATH/executor_runner/libraries"
INCLUDE_DIR_PATH="$EXAMPLES_COREML_DIR_PATH/executor_runner/include"
EXECUTORCH_INCLUDE_DIR_PATH="$COREML_DIR_PATH/runtime/include/executorch"

cd "$EXECUTORCH_ROOT_PATH"

echo "ExecuTorch: Building executor_runner"

echo "ExecuTorch: Removing build directory $CMAKE_BUILD_DIR_PATH"
rm -rf "$CMAKE_BUILD_DIR_PATH"

# Build executorch
echo "ExecuTorch: Building executorch"

cmake -S $EXECUTORCH_ROOT_PATH \
      -B $CMAKE_BUILD_DIR_PATH \
      -DCMAKE_BUILD_TYPE="$MODE" \
      -DEXECUTORCH_BUILD_DEVTOOLS=ON \
      -Dprotobuf_BUILD_TESTS=OFF \
      --preset macos

cmake --build $CMAKE_BUILD_DIR_PATH \
      -j$(sysctl -n hw.ncpu) \
      --config "$MODE" \
      --target coremldelegate \
      --target etdump \
      --target flatccrt

# Copy CoreML delegate headers
echo "ExecuTorch: Copying headers"
echo $EXECUTORCH_INCLUDE_DIR_PATH
rm -rf "$INCLUDE_DIR_PATH"
mkdir -p "$INCLUDE_DIR_PATH"
#Copy ExecuTorch headers
mkdir -p "$EXECUTORCH_INCLUDE_DIR_PATH"
find extension \( -name "*.h" -o -name "*.hpp" \) -exec rsync -R '{}' "$EXECUTORCH_INCLUDE_DIR_PATH" \;
find runtime \( -name "*.h" -o -name "*.hpp" \) -exec rsync -R '{}' "$EXECUTORCH_INCLUDE_DIR_PATH" \;
find util \( -name "*.h" -o -name "*.hpp" \) -exec rsync -R '{}' "$EXECUTORCH_INCLUDE_DIR_PATH" \;
find devtools \( -name "*.h" -o -name "*.hpp" \) -exec rsync -R '{}' "$EXECUTORCH_INCLUDE_DIR_PATH" \;
cp -rf "$COREML_DIR_PATH/runtime/include/" "$INCLUDE_DIR_PATH"

# Copy required libraries
echo "ExecuTorch: Copying libraries"
mkdir -p "$LIBRARIES_DIR_PATH"
find "$CMAKE_BUILD_DIR_PATH/" -name 'libexecutorch.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libexecutorch.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libexecutorch_core.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libexecutorch_core.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libprotobuf-lite.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libprotobuf-lite.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libprotobuf-lited.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libprotobuf-lite.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libetdump.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libetdump.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libcoreml_util.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libcoreml_util.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libcoreml_inmemoryfs.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libcoreml_inmemoryfs.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libcoremldelegate.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libcoremldelegate.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libportable_ops_lib.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libportable_ops_lib.a"  \;
find "$CMAKE_BUILD_DIR_PATH/" -name 'libportable_kernels.a' -exec cp -f "{}" "$LIBRARIES_DIR_PATH/libportable_kernels.a"  \;
cp -f "$CMAKE_BUILD_DIR_PATH/third-party/flatcc_ep/lib/libflatccrt.a" "$LIBRARIES_DIR_PATH/libflatccrt.a"

# Build the runner
echo "ExecuTorch: Building runner"
XCODE_WORKSPACE_DIR_PATH="$EXAMPLES_COREML_DIR_PATH/executor_runner"
XCODE_BUILD_DIR_PATH="$EXAMPLES_COREML_DIR_PATH/xcode-build"

xcodebuild build -workspace "$XCODE_WORKSPACE_DIR_PATH/coreml_executor_runner.xcworkspace" -scheme coreml_executor_runner BUILD_DIR="$XCODE_BUILD_DIR_PATH"

if [[ -z "${COREML_EXECUTOR_RUNNER_OUT_DIR:-}" ]]; then
    COREML_EXECUTOR_RUNNER_OUT_DIR=$(pwd)
elif [[ ! -d "${COREML_EXECUTOR_RUNNER_OUT_DIR}" ]]; then
    mkdir -p "${COREML_EXECUTOR_RUNNER_OUT_DIR}"
fi
cp -f "$XCODE_BUILD_DIR_PATH/DEBUG/coreml_executor_runner" "${COREML_EXECUTOR_RUNNER_OUT_DIR}"
echo "created ${COREML_EXECUTOR_RUNNER_OUT_DIR}/coreml_executor_runner"

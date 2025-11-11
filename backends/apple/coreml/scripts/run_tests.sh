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

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../../../")
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"

cd "$EXECUTORCH_ROOT_PATH"

# Run the test
echo "ExecuTorch: Running executorchcoreml_tests"
XCODE_WORKSPACE_DIR_PATH="$COREML_DIR_PATH/runtime/workspace"
XCODE_BUILD_DIR_PATH="$COREML_DIR_PATH/xcode-build"

xcodebuild test -workspace "$XCODE_WORKSPACE_DIR_PATH/executorchcoreml.xcworkspace" -scheme executorchcoreml_tests BUILD_DIR="$XCODE_BUILD_DIR_PATH"

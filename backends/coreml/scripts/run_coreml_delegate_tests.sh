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

# Run the test
echo "Executorch: Running executorchcoreml_tests"
XCODE_WORKSPACE_DIR_PATH="$COREML_DIR_PATH/runtime/runner"
XCODE_BUILD_DIR_PATH="$COREML_DIR_PATH/xcode-test-build"

xcodebuild test -workspace "$XCODE_WORKSPACE_DIR_PATH/executorchcoreml.xcworkspace" -scheme executorchcoreml_tests BUILD_DIR="$XCODE_BUILD_DIR_PATH"
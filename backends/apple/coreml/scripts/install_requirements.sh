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

red=`tput setaf 1`
green=`tput setaf 2`

EXECUTORCH_ROOT_PATH=$(realpath "$SCRIPT_DIR_PATH/../../../../")
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"
COREMLTOOLS_DIR_PATH="$COREML_DIR_PATH/third-party/coremltools"
PROTOBUF_FILES_DIR_PATH="$COREMLTOOLS_DIR_PATH/build/mlmodel/format/"

cd "$EXECUTORCH_ROOT_PATH"

rm -rf "$COREML_DIR_PATH/third-party"
mkdir "$COREML_DIR_PATH/third-party"

echo "${green}ExecuTorch: Cloning nlohmann."
git clone https://github.com/nlohmann/json.git "$COREML_DIR_PATH/third-party/nlohmann_json"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone nlohmann."
    exit 1
fi

echo "${green}ExecuTorch: Copying protobuf files."
mkdir -p "$COREML_DIR_PATH/runtime/sdk/format/"
cp -rf "$PROTOBUF_FILES_DIR_PATH" "$COREML_DIR_PATH/runtime/sdk/format/"

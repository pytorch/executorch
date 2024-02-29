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

cd "$EXECUTORCH_ROOT_PATH"

# clone and install coremltools
if [ -d "/tmp/coremltools" ]; then
    rm -rf "/tmp/coremltools"
fi

echo "${green}ExecuTorch: Cloning coremltools."
git clone "https://github.com/apple/coremltools.git" /tmp/coremltools
cd /tmp/coremltools
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone coremltools."
    exit 1
fi

echo "${green}ExecuTorch: Installing coremltools dependencies."
pip install -r /tmp/coremltools/reqs/build.pip
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to install coremltools dependencies."
    exit 1
fi

mkdir /tmp/coremltools/build
cmake -S /tmp/coremltools/ -B /tmp/coremltools/build
cmake --build /tmp/coremltools/build --parallel

echo "${green}ExecuTorch: Installing coremltools."
pip install /tmp/coremltools
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to install coremltools."
    exit 1
fi

cd "$EXECUTORCH_ROOT_PATH"

rm -rf "$COREML_DIR_PATH/third-party"
mkdir "$COREML_DIR_PATH/third-party"

echo "${green}ExecuTorch: Cloning ios-cmake."
git clone https://github.com/leetal/ios-cmake.git "$COREML_DIR_PATH/third-party/ios-cmake"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone ios-cmake."
    exit 1
fi

echo "${green}ExecuTorch: Cloning nlohmann."
git clone https://github.com/nlohmann/json.git "$COREML_DIR_PATH/third-party/nlohmann_json"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to clone nlohmann."
    exit 1
fi

echo "${green}ExecuTorch: Installing inmemoryfs extension."
pip install "$COREML_DIR_PATH/runtime/inmemoryfs"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to install inmemoryfs extension."
    exit 1
fi

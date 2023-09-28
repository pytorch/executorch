#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script to sanity check CoreML backend.

SCRIPT_DIR_PATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

EXECUTORCH_ROOT_PATH="$SCRIPT_DIR_PATH/../../../../"
COREML_DIR_PATH="$EXECUTORCH_ROOT_PATH/backends/apple/coreml"
COREML_EXAMPLES_DIR_PATH="$EXECUTORCH_ROOT_PATH/examples/apple/coreml"
TEST_ENV_NAME=".coreml-test-env"
TEST_MODEL_PATH="$COREML_DIR_PATH/runtime/test/models/mv3_coreml_all.pte"

red=`tput setaf 1`
green=`tput setaf 2`

cd "$EXECUTORCH_ROOT_PATH"

echo "${green}ExecuTorch: Updating git submmodules"
git submodule sync
git submodule update --init

STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to update git submodules."
    exit 1
fi

echo "${green}ExecuTorch: Creating virtual environment."

function cleanup {
    echo "${green}ExecuTorch: Deactivating virtual environment"
    deactivate
    rm -rf "$TEST_ENV_NAME"
}

python3 -m venv "$TEST_ENV_NAME"
source "$TEST_ENV_NAME/bin/activate"
trap cleanup EXIT

STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to create virtual environment."
    exit 1
fi

FLATBUFFERS_PATH="$EXECUTORCH_ROOT_PATH/third-party/flatbuffers/cmake-out"
export PATH="${FLATBUFFERS_PATH}:${PATH}"

echo "${green}ExecuTorch: Installing ExecuTorch requirements"
source "$EXECUTORCH_ROOT_PATH/install_requirements.sh"

echo "${green}ExecuTorch: Installing CoreML requirements"
source "$COREML_DIR_PATH/scripts/install_requirements_internal.sh"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to install dependencies required by CoreML backend."
    exit 1
fi

cd "$EXECUTORCH_ROOT_PATH"

# Build CoreML tests
echo "${green}ExecuTorch: Building CoreML tests"
source "$COREML_DIR_PATH/scripts/build_tests.sh"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to build CoreML tests."
    exit 1
fi

# Run CoreML tests
echo "${green}ExecuTorch: Running CoreML tests"
source "$COREML_DIR_PATH/scripts/run_tests.sh"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to run CoreML tests."
    exit 1
fi

# Build CoreML runner, this also builds the CoreML delegate
echo "${green}ExecuTorch: Building CoreML executor."
cd "$EXECUTORCH_ROOT_PATH"
source "$COREML_EXAMPLES_DIR_PATH/scripts/build_executor_runner.sh"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to build executor."
    exit 1
fi

# Run CoreML runner
echo "${green}ExecuTorch: Verifying mv3 model"
$("$EXECUTORCH_ROOT_PATH"/coreml_executor_runner --model_path "$TEST_MODEL_PATH")
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "${red}ExecuTorch: Failed to run runner."
    exit 1
fi

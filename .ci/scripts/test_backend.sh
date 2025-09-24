#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SUITE=$1
FLOW=$2
ARTIFACT_DIR=$3

REPORT_FILE="$ARTIFACT_DIR/test-report-$FLOW-$SUITE.json"

echo "Running backend test job for suite $SUITE, flow $FLOW."
echo "Saving job artifacts to $ARTIFACT_DIR."

eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

if [[ "$(uname)" == "Darwin" ]]; then
    ${CONDA_RUN} --no-capture-output pip install awscli==1.37.21
    IS_MACOS=1
    CONDA_PREFIX="${CONDA_RUN} --no-capture-output"
else
    IS_MACOS=0
    CONDA_PREFIX=""
fi

export PYTHON_EXECUTABLE=python

# CMake options to use, in addition to the defaults.
EXTRA_BUILD_ARGS=""

if [[ "$FLOW" == *qnn* ]]; then
    # Setup QNN sdk and deps - note that this is a bit hacky due to the nature of the
    # Qualcomm build. TODO (gjcomer) Clean this up once the QNN pybinding integration is
    # cleaned up.
    PYTHON_EXECUTABLE=python bash .ci/scripts/setup-linux.sh --build-tool cmake
    PYTHON_EXECUTABLE=python source .ci/scripts/build-qnn-sdk.sh
    QNN_X86_LIB_DIR=`realpath build-x86/lib/`
    export LD_LIBRARY_PATH"=$QNN_X86_LIB_DIR:$QNN_SDK_ROOT/lib/x86_64-linux-clang/:${LD_LIBRARY_PATH:-}"

    # TODO Get SDK root from install scripts
    EXTRA_BUILD_ARGS+=" -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=$QNN_SDK_ROOT"
fi

if [[ "$FLOW" == *vulkan* ]]; then
    # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate.
    source .ci/scripts/setup-vulkan-linux-deps.sh

    EXTRA_BUILD_ARGS+=" -DEXECUTORCH_BUILD_VULKAN=ON"
fi

if [[ "$FLOW" == *arm* ]]; then
    # Setup ARM deps.
    .ci/scripts/setup-arm-baremetal-tools.sh
fi

if [[ $IS_MACOS -eq 1 ]]; then
    SETUP_SCRIPT=.ci/scripts/setup-macos.sh
else
    SETUP_SCRIPT=.ci/scripts/setup-linux.sh
fi
${CONDA_PREFIX} CMAKE_ARGS="$EXTRA_BUILD_ARGS" $SETUP_SCRIPT --build-tool cmake --build-mode Release --editable true

EXIT_CODE=0
${CONDA_PREFIX} pytest -c /dev/nul -n auto backends/test/suite/$SUITE/ -m flow_$FLOW --json-report --json-report-file="$REPORT_FILE" || EXIT_CODE=$?
# Generate markdown summary.
${CONDA_PREFIX} python -m executorch.backends.test.suite.generate_markdown_summary_json "$REPORT_FILE" > ${GITHUB_STEP_SUMMARY:-"step_summary.md"} --exit-code $EXIT_CODE

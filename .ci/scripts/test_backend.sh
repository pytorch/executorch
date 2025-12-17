#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
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
    bash .ci/scripts/setup-conda.sh
    eval "$(conda shell.bash hook)"
    CONDA_RUN_CMD="${CONDA_RUN} --no-capture-output"
    ${CONDA_RUN_CMD} pip install awscli==1.37.21
    IS_MACOS=1
else
    CONDA_RUN_CMD=""
    IS_MACOS=0
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
    if [[ "$FLOW" == *vgf* ]]; then
        .ci/scripts/setup-arm-baremetal-tools.sh --enable-mlsdk-deps --install-mlsdk-deps-with-pip
    else
        .ci/scripts/setup-arm-baremetal-tools.sh
    fi
    source examples/arm/arm-scratch/setup_path.sh

    if [[ "$FLOW" == *ethos_u* ]]; then
        # Prepare a test runner binary that can run on the Corstone-3x0 FVPs
        backends/arm/scripts/build_executorch.sh
        backends/arm/test/setup_testing.sh
    fi

    if [[ "$FLOW" == *vgf* ]]; then
        # Prepare a test runner binary for VKML runtime
        backends/arm/test/setup_testing_vkml.sh
    fi
fi

if [[ $IS_MACOS -eq 1 ]]; then
    SETUP_SCRIPT=.ci/scripts/setup-macos.sh
else
    SETUP_SCRIPT=.ci/scripts/setup-linux.sh
fi
CMAKE_ARGS="$EXTRA_BUILD_ARGS" ${CONDA_RUN_CMD} $SETUP_SCRIPT --build-tool cmake --build-mode Release --editable true

EXIT_CODE=0
${CONDA_RUN_CMD} pytest -c /dev/nul -n auto backends/test/suite/$SUITE/ -m flow_$FLOW --json-report --json-report-file="$REPORT_FILE" || EXIT_CODE=$?
# Generate markdown summary.
${CONDA_RUN_CMD} python -m executorch.backends.test.suite.generate_markdown_summary_json "$REPORT_FILE" > ${GITHUB_STEP_SUMMARY:-"step_summary.md"} --exit-code $EXIT_CODE

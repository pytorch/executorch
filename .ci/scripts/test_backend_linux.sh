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

echo "Running backend test job for suite $SUITE, flow $FLOW."
echo "Saving job artifacts to $ARTIFACT_DIR."

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

# CMake options to use, in addition to the defaults.
EXTRA_BUILD_ARGS=""

if [[ "$FLOW" =~ ".*qualcomm.*"]]; then
    # Setup QNN sdk and deps
    ./install_requirements.sh --use-pt-pinned-commit
    PYTHON_EXECUTABLE=python bash .ci/scripts/setup-qnn-deps.sh
    PYTHON_EXECUTABLE=python bash .ci/scripts/build-qnn-sdk.sh

    EXTRA_BUILD_ARGS+=" -DEXECUTORCH_BUILD_QNN=ON"
fi

if [[ "$FLOW" =~ ".*vulkan.*"]]; then
    # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
    source .ci/scripts/setup-vulkan-linux-deps.sh
    
    EXTRA_BUILD_ARGS+=" -DEXECUTORCH_BUILD_VULKAN=ON"
fi

# We need the runner to test the built library.
PYTHON_EXECUTABLE=python CMAKE_ARGS="$EXTRA_BUILD_ARGS" .ci/scripts/setup-linux.sh --build-tool cmake --build-mode Release

python -m executorch.backends.test.suite.runner $SUITE --flow $FLOW --report "$ARTIFACT_DIR/test_results.csv"

#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

SUITE=$1
FLOW=$2

echo "Running backend test job for suite $SUITE, flow $FLOW."

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

# Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
#source .ci/scripts/setup-vulkan-linux-deps.sh

# We need the runner to test the built library.
PYTHON_EXECUTABLE=python .ci/scripts/setup-linux.sh --build-tool cmake --build-mode Release

python -m executorch.backends.test.suite.runner $SUITE --flow $FLOW --report test_results.csv

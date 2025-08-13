#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

read -r SUITE FLOW < <(parse_args "$@")

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

# Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
#source .ci/scripts/setup-vulkan-linux-deps.sh

# We need the runner to test the built library.
.ci/scripts/setup-linux.sh "cmake" "release" "false"

python -m executorch.backends.test.suite.runner $SUITE --flow $FLOW --report test_results.csv

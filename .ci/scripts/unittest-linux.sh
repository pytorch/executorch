#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

read -r BUILD_TOOL BUILD_MODE EDITABLE < <(parse_args "$@")

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
    source .ci/scripts/setup-vulkan-linux-deps.sh

    PYTHON_EXECUTABLE=python \
    CMAKE_ARGS="-DEXECUTORCH_BUILD_PYBIND=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON" \
    .ci/scripts/setup-linux.sh "$@"

    # Install llama3_2_vision dependencies.
    PYTHON_EXECUTABLE=python ./examples/models/llama3_2_vision/install_requirements.sh

    .ci/scripts/unittest-linux-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    # Removing this breaks sccache in the Buck build, apparently
    # because TMPDIR gets messed up? Please feel free to fix this and
    # speed up this CI job!
    PYTHON_EXECUTABLE=python \
    .ci/scripts/setup-linux.sh "$@"

    .ci/scripts/unittest-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

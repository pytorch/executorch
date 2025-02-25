#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

BUILD_TOOL=$1
if [[ $BUILD_TOOL =~ ^(cmake|buck2)$ ]]; then
    echo "Running unittests for ${BUILD_TOOL} ..."
else
  echo "Missing build tool (require buck2 or cmake), exiting..."
  exit 1
fi

BUILD_MODE=$2
if [[ "${BUILD_MODE:-}" =~ ^(Debug|Release)$ ]]; then
    echo "Running tests in build mode ${BUILD_MODE} ..."
else
    echo "Unsupported build mode ${BUILD_MODE}, options are Debug or Release."
    exit 1
fi

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    # Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
    source .ci/scripts/setup-vulkan-linux-deps.sh

    PYTHON_EXECUTABLE=python \
    EXECUTORCH_BUILD_PYBIND=ON \
    CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON" \
    .ci/scripts/setup-linux.sh "$BUILD_TOOL" "$BUILD_MODE"

    # Install llama3_2_vision dependencies.
    PYTHON_EXECUTABLE=python ./examples/models/llama3_2_vision/install_requirements.sh

    .ci/scripts/unittest-linux-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    # XXX: check whether this is sufficient to unbreak sccache
    PYTHON_EXECUTABLE=python \
    .ci/scripts/setup-linux.sh "$BUILD_TOOL" "$BUILD_MODE"

    .ci/scripts/unittest-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

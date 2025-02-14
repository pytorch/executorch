#!/bin/bash
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

# The generic Linux job chooses to use base env, not the one setup by the image
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

# Setup swiftshader and Vulkan SDK which are required to build the Vulkan delegate
source .ci/scripts/setup-vulkan-linux-deps.sh

PYTHON_EXECUTABLE=python \
EXECUTORCH_BUILD_PYBIND=ON \
CMAKE_ARGS="-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON" \
.ci/scripts/setup-linux.sh "$BUILD_TOOL"

# Install llama3_2_vision dependencies.
PYTHON_EXECUTABLE=python ./examples/models/llama3_2_vision/install_requirements.sh

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    .ci/scripts/unittest-linux-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    .ci/scripts/unittest-linux-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

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
if [[ $BUILD_MODE =~ ^(Debug|Release)$ ]]; then
    echo "Running tests in build mode ${BUILD_MODE} ..."
else
    echo "Unsupported build mode ${BUILD_MODE}, options are Debug or Release."
    exit 1
fi

bash .ci/scripts/setup-conda.sh
eval "$(conda shell.bash hook)"

# Create temp directory for sccache shims
export TMP_DIR=$(mktemp -d)
export PATH="${TMP_DIR}:$PATH"
trap 'rm -rfv ${TMP_DIR}' EXIT

# Setup MacOS dependencies as there is no Docker support on MacOS atm
PYTHON_EXECUTABLE=python \
EXECUTORCH_BUILD_PYBIND=ON \
CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON" \
${CONDA_RUN} --no-capture-output \
.ci/scripts/setup-macos.sh "${BUILD_TOOL}" "${BUILD_MODE}"

# Install llama3_2_vision dependencies.
PYTHON_EXECUTABLE=python ./examples/models/llama3_2_vision/install_requirements.sh

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    .ci/scripts/unittest-macos-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    .ci/scripts/unittest-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

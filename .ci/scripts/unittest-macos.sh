#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

parse_args "$@"

bash .ci/scripts/setup-conda.sh
eval "$(conda shell.bash hook)"

# Create temp directory for sccache shims
export TMP_DIR=$(mktemp -d)
export PATH="${TMP_DIR}:$PATH"
trap 'rm -rfv ${TMP_DIR}' EXIT

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    # Setup MacOS dependencies as there is no Docker support on MacOS atm
    PYTHON_EXECUTABLE=python \
    EXECUTORCH_BUILD_PYBIND=ON \
    CMAKE_ARGS="-DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_MPS=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON" \
    ${CONDA_RUN} --no-capture-output \
    .ci/scripts/setup-macos.sh "$@"

    # Install llama3_2_vision dependencies.
    PYTHON_EXECUTABLE=python \
    ${CONDA_RUN} --no-capture-output \
    ./examples/models/llama3_2_vision/install_requirements.sh

    .ci/scripts/unittest-macos-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    .ci/scripts/unittest-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

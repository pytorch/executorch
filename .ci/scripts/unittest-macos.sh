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

bash .ci/scripts/setup-conda.sh
eval "$(conda shell.bash hook)"

# Create temp directory for sccache shims
export TMP_DIR=$(mktemp -d)
export PATH="${TMP_DIR}:$PATH"
trap 'rm -rfv ${TMP_DIR}' EXIT

# Enable sanitizers for Debug builds
if [[ "$BUILD_MODE" == "Debug" ]]; then
    export EXECUTORCH_USE_SANITIZER=ON
fi

# Setup MacOS dependencies as there is no Docker support on MacOS atm
# We need the runner to test the built library.
PYTHON_EXECUTABLE=python \
CMAKE_ARGS="-DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON -DEXECUTORCH_BUILD_TESTS=ON" \
${CONDA_RUN} --no-capture-output \
.ci/scripts/setup-macos.sh "$@"

if [[ "$BUILD_TOOL" == "cmake" ]]; then
    # Install llama3_2_vision dependencies.
    PYTHON_EXECUTABLE=python \
    ${CONDA_RUN} --no-capture-output \

    .ci/scripts/unittest-macos-cmake.sh
elif [[ "$BUILD_TOOL" == "buck2" ]]; then
    .ci/scripts/unittest-buck2.sh
    # .ci/scripts/unittest-macos-buck2.sh
else
    echo "Unknown build tool $BUILD_TOOL"
    exit 1
fi

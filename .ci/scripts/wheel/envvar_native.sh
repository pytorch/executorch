# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export NATIVE_WHEEL_ROOT="${GITHUB_WORKSPACE}/${REPOSITORY}"
export NATIVE_WHEEL_DEPS="${GITHUB_WORKSPACE}/native-wheel-deps"
export CMAKE_PREFIX_PATH="${NATIVE_WHEEL_DEPS}/libzip:${NATIVE_WHEEL_DEPS}:${NATIVE_WHEEL_DEPS}/shaderc${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
export FLATC_EXECUTABLE="${NATIVE_WHEEL_DEPS}/flatbuffers/flatc"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DFLATC_EXECUTABLE=${FLATC_EXECUTABLE} -DGLSLC_PATH=${NATIVE_WHEEL_DEPS}/shaderc/bin/glslc -DEXECUTORCH_NATIVE_EXTRA_LICENSES=${NATIVE_WHEEL_DEPS}/libzip-1.11.4/LICENSE"
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:+${PIP_EXTRA_INDEX_URL} }https://pypi.org/simple"

# Nova sources this hook before invoking setup.py, allowing a subdirectory build.
if [[ "$(pwd)" == "${NATIVE_WHEEL_ROOT}" ]]; then
    cd "${NATIVE_WHEEL_ROOT}/backends/native/python" || exit 1
fi

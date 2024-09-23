#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

unset CMAKE_PREFIX_PATH
git submodule sync
git submodule update --init
./install_requirements.sh

rm -rf cmake-out

STEPWISE_BUILD=false

if $STEPWISE_BUILD; then
    echo "Building ExecuTorch"
    cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake  \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CPUINFO=OFF \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=OFF \
        -DFLATC_EXECUTABLE="$(which flatc)" \
        -Bcmake-out .

    echo "Building any Cadence-specific binaries on top"
    cmake -DBUCK2="$BUCK" \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_HOST_TARGETS=ON \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=ON \
        -DFLATC_EXECUTABLE="$(which flatc)" \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF \
        -DPYTHON_EXECUTABLE=python3 \
        -DEXECUTORCH_NNLIB_OPT=ON \
        -DEXECUTORCH_BUILD_GFLAGS=ON \
        -DHAVE_FNMATCH_H=OFF \
        -Bcmake-out/backends/cadence \
        backends/cadence
    cmake --build cmake-out/backends/cadence  -j16
else
    echo "Building Cadence toolchain with ExecuTorch packages"
    cmake_prefix_path="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
    cmake -DBUCK2="$BUCK" \
        -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_HOST_TARGETS=ON \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=OFF \
        -DFLATC_EXECUTABLE="$(which flatc)" \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DBUILD_EXECUTORCH_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=OFF \
        -DPYTHON_EXECUTABLE=python3 \
        -DEXECUTORCH_NNLIB_OPT=ON \
        -DEXECUTORCH_BUILD_GFLAGS=ON \
        -DHAVE_FNMATCH_H=OFF \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_BUILD_CPUINFO=OFF \
        -Bcmake-out
    cmake --build cmake-out --target install --config Release -j16
fi

echo "Run simple model to verify cmake build"
python3 -m examples.portable.scripts.export --model_name="add"
xt-run --turbo cmake-out/executor_runner  --model_path=add.pte

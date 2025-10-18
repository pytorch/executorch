#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

unset CMAKE_PREFIX_PATH
unset XTENSA_CORE
export XTENSA_CORE=FCV_FG3GP
git submodule sync
git submodule update --init
./backends/cadence/install_requirements.sh
./install_executorch.sh

rm -rf cmake-out

STEPWISE_BUILD=false

if $STEPWISE_BUILD; then
    echo "Building ExecuTorch"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake  \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CPUINFO=OFF \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=OFF \
        -DHAVE_FNMATCH_H=OFF \
        -Bcmake-out .

    echo "Building any Cadence-specific binaries on top"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
        -DCMAKE_TOOLCHAIN_FILE=/home/zonglinpeng/ws/zonglinpeng/executorch/backends/cadence/cadence.cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CADENCE=ON \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
        -DPYTHON_EXECUTABLE=python3 \
        -DEXECUTORCH_FUSION_G3_OPT=ON \
        -DHAVE_FNMATCH_H=OFF \
        -Bcmake-out/backends/cadence \
        backends/cadence
    cmake --build cmake-out/backends/cadence  -j8
else
    echo "Building Cadence toolchain with ExecuTorch packages"
    cmake_prefix_path="${PWD}/cmake-out/lib/cmake/ExecuTorch;${PWD}/cmake-out/third-party/gflags"
    CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
        -DCMAKE_PREFIX_PATH="${cmake_prefix_path}" \
        -DHAVE_SYS_STAT_H=ON \
        -DCMAKE_TOOLCHAIN_FILE=./backends/cadence/cadence.cmake \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
        -DEXECUTORCH_BUILD_PTHREADPOOL=OFF \
        -DEXECUTORCH_BUILD_CPUINFO=OFF \
        -DEXECUTORCH_BUILD_CADENCE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_ENABLE_LOGGING=ON \
        -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON \
        -DEXECUTORCH_USE_DL=OFF \
        -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
        -DEXECUTORCH_BUILD_KERNELS_LLM=OFF \
        -DPYTHON_EXECUTABLE=python3 \
        -DEXECUTORCH_FUSION_G3_OPT=ON \
        -DHAVE_FNMATCH_H=OFF \
        -Bcmake-out
    cmake --build cmake-out --target install --config Release -j8
fi

echo "Run simple model to verify cmake build"
python3 -m examples.portable.scripts.export --model_name="add"
xt-run --turbo cmake-out/executor_runner  --model_path=add.pte

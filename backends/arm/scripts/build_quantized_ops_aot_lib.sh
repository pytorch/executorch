#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Needs to be run from exeuctorch root.
# Optional parameter: 1: build_type= "Release" | "Debug" | "RelWithDebInfo"

build_type="Release"

build_type=${1:-$build_type}

echo "--------------------------------------------------------------------------------"
echo "Build .so library to register quant ops with AoT flow ${build_type} into '$(echo $(pwd))/cmake-out-aot-lib'"
echo "--------------------------------------------------------------------------------"

# Since we only want to build the quantized_aot lib in the specified folder,
# we want exactly the configuration set below and deleting the cache is OK.
rm -f cmake-out-aot-lib/CMakeCache.txt

CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
    -DCMAKE_BUILD_TYPE=${build_type}            \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON      \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
    -Bcmake-out-aot-lib                         \
    .

cmake --build cmake-out-aot-lib --parallel -- quantized_ops_aot_lib

#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Optional parameter: 1: build_type= "Release" | "Debug" | "RelWithDebInfo"

set -eu
script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})

build_type="Release"
et_build_root="${et_root_dir}"

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --et_build_root=<FOLDER>  Build output root folder to use, defaults to ${et_build_root}"
    echo "  --build_type=<TYPE>       Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      *)
      ;;
    esac
done

et_build_dir=${et_build_root}/cmake-out-aot-lib

cd "${et_root_dir}"

echo "--------------------------------------------------------------------------------"
echo "Build quantized_ops_aot_lib library to register quant ops with AoT flow ${build_type} into '${et_build_dir}'"
echo "--------------------------------------------------------------------------------"

# Since we only want to build the quantized_aot lib in the specified folder,
# we want exactly the configuration set below and deleting the cache is OK.
rm -f ${et_build_dir}/CMakeCache.txt

CXXFLAGS="-fno-exceptions -fno-rtti" cmake \
    -DCMAKE_BUILD_TYPE=${build_type}            \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON      \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
    -B${et_build_dir}                         \
    .

cmake --build ${et_build_dir} --parallel -- quantized_ops_aot_lib

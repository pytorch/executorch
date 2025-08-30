#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Optional parameter:
# --build_type= "Release" | "Debug" | "RelWithDebInfo"
# --etdump      build with devtools-etdump support

set -eu

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
et_root_dir=$(cd ${script_dir}/../../.. && pwd)
et_root_dir=$(realpath ${et_root_dir})
toolchain=arm-none-eabi-gcc
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."

et_build_root="${et_root_dir}/arm_test"
build_type="Release"
build_devtools=OFF
build_with_etdump=OFF

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --et_build_root=<FOLDER>  Build output root folder to use, defaults to ${et_build_root}"
    echo "  --build_type=<TYPE>       Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --devtools                Build Devtools libs"
    echo "  --etdump                  Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    echo "  --toolchain=<TOOLCHAIN>   Toolchain can be specified (e.g. bare metal as arm-none-eabi-gcc or zephyr as arm-zephyr-eabi-gcc Default: ${toolchain}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      --devtools) build_devtools=ON ;;
      --etdump) build_with_etdump=ON ;;
      --toolchain=*) toolchain="${arg#*=}";;
      *)
      ;;
    esac
done

if [[ ${toolchain} == "arm-none-eabi-gcc" ]]; then
    toolchain_cmake=${et_root_dir}/examples/arm/ethos-u-setup/${toolchain}.cmake
elif [[ ${toolchain} == "arm-zephyr-eabi-gcc" ]]; then
    toolchain_cmake=${et_root_dir}/examples/zephyr/x86_64-linux-arm-zephyr-eabi-gcc.cmake
else
    echo "Error: Invalid toolchain selection, provided: ${toolchain}"
    echo "    Valid options are {arm-none-eabi-gcc, arm-zephyr-eabi-gcc}"
    exit 1;
fi
toolchain_cmake=$(realpath ${toolchain_cmake})

# Source the tools
# This should be prepared by the setup.sh
[[ -f ${setup_path_script} ]] \
    || { echo "Missing ${setup_path_script}. ${_setup_msg}"; exit 1; }

source ${setup_path_script}

et_build_dir="${et_build_root}/cmake-out"

set -x
cd "${et_root_dir}"

( set +x ;
    echo "--------------------------------------------------------------------------------" ;
    echo "Build ExecuTorch target libs ${build_type} into '${et_build_dir}'" ;
    echo "--------------------------------------------------------------------------------" )

# Build
cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_cmake} \
-DCMAKE_BUILD_TYPE=Release \
-DEXECUTORCH_BUILD_DEVTOOLS=$build_devtools \
-DEXECUTORCH_BUILD_ARM_ETDUMP=$build_with_etdump \
--preset arm-baremetal -B${et_build_dir}

cmake --build ${et_build_dir} -j$(nproc) --target install --config ${build_type} --

set +x

echo "[$(basename $0)] Generated static libraries for ExecuTorch:"
find ${et_build_dir} -name "*.a" -exec ls -al {} \;

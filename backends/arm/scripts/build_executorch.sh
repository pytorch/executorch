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
toolchain_cmake=${script_dir}/../../../examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake
toolchain_cmake=$(realpath ${toolchain_cmake})
setup_path_script=${et_root_dir}/examples/arm/ethos-u-scratch/setup_path.sh
_setup_msg="please refer to ${et_root_dir}/examples/arm/setup.sh to properly install necessary tools."

et_build_root="${et_root_dir}/arm_test"
build_type="Release"
build_devtools=false
build_with_etdump=false

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --et_build_root=<FOLDER>  Build output root folder to use, defaults to ${et_build_root}"
    echo "  --build_type=<TYPE>       Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --devtools                Build Devtools libs"
    echo "  --etdump                  Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      --devtools) build_devtools=true ;;
      --etdump) build_with_etdump=true ;;
      *)
      ;;
    esac
done

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

build_devtools_flags=" -DEXECUTORCH_BUILD_DEVTOOLS=OFF "
if [ "$build_devtools" = true ] ; then
    build_devtools_flags=" -DEXECUTORCH_BUILD_DEVTOOLS=ON "
fi

build_with_etdump_flags=" -DEXECUTORCH_ENABLE_EVENT_TRACER=OFF "
if [ "$build_with_etdump" = true ] ; then
    # Add DevTools flags use in the Target build below
    build_with_etdump_flags="-DEXECUTORCH_BUILD_DEVTOOLS=ON                    \
                            -DEXECUTORCH_ENABLE_EVENT_TRACER=ON               \
                            -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=OFF      \
                            -DFLATCC_ALLOW_WERROR=OFF "
fi

echo "Building with Devtools: ${build_devtools_flags} ${build_with_etdump_flags}"


# Build
cmake                                                 \
    -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
    -DCMAKE_BUILD_TYPE=${build_type}                  \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
    -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON           \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
    -DEXECUTORCH_BUILD_CORTEX_M=ON                    \
    -DEXECUTORCH_ENABLE_LOGGING=ON                    \
    ${build_devtools_flags}                           \
    ${build_with_etdump_flags}                        \
    -B"${et_build_dir}"                               \
    "${et_root_dir}"

echo "[$(basename $0)] Configured CMAKE"

cmake --build ${et_build_dir} -j$(nproc) --target install --config ${build_type} --

set +x

echo "[$(basename $0)] Generated static libraries for ExecuTorch:"
find ${et_build_dir} -name "*.a" -exec ls -al {} \;

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



et_build_root="${et_root_dir}/arm_test"
build_type="Release"
build_with_etdump=false


help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --et_build_root=<FOLDER>  Build output root folder to use, defaults to ${et_build_root}"
    echo "  --build_type=<TYPE>       Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --etdump                  Adds Devtools etdump support to track timing, etdump area will be base64 encoded in the log"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      --etdump) build_with_etdump=true ;;
      *)
      ;;
    esac
done

et_build_dir="${et_build_root}/cmake-out"
et_build_host_dir=${et_build_root}/cmake-out-host-tools

set -x
cd "${et_root_dir}"

build_with_etdump_flags=""
if [ "$build_with_etdump" = true ] ; then
    ( set +x ;
        echo "--------------------------------------------------------------------------------" ;
        echo "Build ExecuTorch Libraries host flatcc bin ${build_type} into ${et_build_host_dir} - ${et_build_host_dir}/bin/flatcc" ;
        echo "--------------------------------------------------------------------------------" )


    # Build host flatcc bin
    # This is a way to work around that the flatcc executable get build for target (e.g. Arm) later
    # and get replaced. flatcc is a tool used on the host for etdump and BundleIO handling.
    # The way to solve this is to generate it once for the host, then copy it to ${et_build_host_dir}/bin
    # and later point that out with -DFLATCC_EXECUTABLE=${et_build_host_dir}/bin/flatcc later.
    mkdir -p ${et_build_host_dir}
    cmake                                                 \
        -DCMAKE_INSTALL_PREFIX=${et_build_host_dir}       \
        -DCMAKE_BUILD_TYPE=${build_type}                  \
        -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
        -DEXECUTORCH_ENABLE_LOGGING=ON                    \
        -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON           \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON                    \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON               \
        -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=ON      \
        -DFLATCC_ALLOW_WERROR=OFF                         \
        -DFLATC_EXECUTABLE="$(which flatc)"               \
        -B"${et_build_host_dir}"                          \
        "${et_root_dir}"

    # Copy host flatcc excutable to it's saved when we build for target (Arm) later
    mkdir -p ${et_build_host_dir}/bin
    cp third-party/flatcc/bin/flatcc ${et_build_host_dir}/bin

    # Add DevTools flags use in the Target build below
    build_with_etdump_flags="-DEXECUTORCH_BUILD_DEVTOOLS=ON                    \
                                -DEXECUTORCH_ENABLE_EVENT_TRACER=ON               \
                                -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=OFF     \
                                -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=OFF      \
                                -DFLATCC_ALLOW_WERROR=OFF                         \
                                -DFLATCC_EXECUTABLE=${et_build_host_dir}/bin/flatcc "
    echo "build_with_etdump_flags=$build_with_etdump_flags"
fi

( set +x ;
    echo "--------------------------------------------------------------------------------" ;
    echo "Build ExecuTorch target libs ${build_type} into '${et_build_dir}'" ;
    echo "--------------------------------------------------------------------------------" )

# Build
cmake                                                 \
    -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
    -DCMAKE_BUILD_TYPE=${build_type}                  \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF            \
    -DEXECUTORCH_BUILD_ARM_BAREMETAL=ON               \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON           \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON       \
    -DEXECUTORCH_ENABLE_LOGGING=ON                    \
    ${build_with_etdump_flags}                        \
    -DFLATC_EXECUTABLE="$(which flatc)"               \
    -B"${et_build_dir}"                               \
    "${et_root_dir}"

echo "[$(basename $0)] Configured CMAKE"

cmake --build ${et_build_dir} --parallel --target install --config ${build_type} --

set +x

echo "[$(basename $0)] Generated static libraries for ExecuTorch:"
find ${et_build_dir} -name "*.a" -exec ls -al {} \;

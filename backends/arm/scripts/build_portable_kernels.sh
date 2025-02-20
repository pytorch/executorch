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
portable_kernels="aten::_softmax.out"

help() {
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  --et_build_root=<FOLDER>   Build output root folder to use, defaults to ${et_build_root}"
    echo "  --build_type=<TYPE>        Build with Release, Debug or RelWithDebInfo, default is ${build_type}"
    echo "  --portable_kernels=<OPS>   Comma separated list of portable (non delagated) kernels to include Default: ${portable_kernels}"
    exit 0
}

for arg in "$@"; do
    case $arg in
      -h|--help) help ;;
      --et_build_root=*) et_build_root="${arg#*=}";;
      --build_type=*) build_type="${arg#*=}";;
      --portable_kernels=*) portable_kernels="${arg#*=}";;
      *)
      ;;
    esac
done

et_build_dir=${et_build_root}/cmake-out

cd "${et_root_dir}"

echo "--------------------------------------------------------------------------------" ;
echo "Build ExecuTorch Libraries ${build_type} portable kernels: ${portable_kernels} into '${et_build_dir}'" ;
echo "--------------------------------------------------------------------------------"

if ! [[ $portable_kernels =~ ^((^|,)aten::[a-zA-Z0-9_]+\.[a-zA-Z0-9_]*out)*$ ]]; then
    echo " ERROR: specified argument --portable_kernels=${portable_kernels}"
    echo "        is in the wrong format please use \"aten::<OP1>.out,aten::<OP2>.out,...\""
    echo "        e.g. \"aten::_softmax.out,aten::add.out\""
    exit 1
fi

set -x

cmake                                                 \
    -DCMAKE_INSTALL_PREFIX=${et_build_dir}            \
    -DCMAKE_BUILD_TYPE=${build_type}                  \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain_cmake}"       \
    -DEXECUTORCH_SELECT_OPS_LIST=${portable_kernels}  \
    -B"${et_build_dir}/examples/arm"                  \
    "${et_root_dir}/examples/arm"

cmake --build "${et_build_dir}/examples/arm" --parallel --config ${build_type} --

set +x

echo "[$(basename $0)] Generated static libraries for ExecuTorch:"
find "${et_build_dir}/examples/arm" -name "*.a" -exec ls -al {} \;

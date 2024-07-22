# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -e

if [[ -z ${QNN_SDK_ROOT} ]]; then
    echo "Please export QNN_SDK_ROOT=/path/to/qnn_sdk"
    exit -1
fi



usage() {
  echo "Usage: Build the aarch64 version of executor runner or the python interface of Qnn Manager"
  echo "First, you need to set the environment variable for QNN_SDK_ROOT"
  echo ", and if you want to build the aarch64 version of executor runner"
  echo ", you need to set ANDROID_NDK_ROOT"
  echo "e.g.: executorch$ ./backends/qualcomm/scripts/build.sh --skip_x86_64"
  exit 1
}


[ "$1" = -h ] && usage

BUILD_X86_64="true"
CMAKE_X86_64="build_x86_64"
BUILD_AARCH64="true"
CMAKE_AARCH64="build_android"
CLEAN="true"

if [ -z PYTHON_EXECUTABLE ]; then
  PYTHON_EXECUTABLE="python3"
fi

if [ -z BUCK2 ]; then
  BUCK2="buck2"
fi

long_options=skip_x86_64,skip_aarch64,no_clean

parsed_args=$(getopt -a --options '' --longoptions $long_options --name "$0" -- "$@")
eval set -- "$parsed_args"


while true ; do
    case "$1" in
        --skip_x86_64) BUILD_X86_64="false"; shift;;
        --skip_aarch64) BUILD_AARCH64="false"; shift;;
        --no_clean) CLEAN="false"; shift;;
        --) shift; break;;
    esac
done

PRJ_ROOT="$( cd "$(dirname "$0")/../../.." ; pwd -P)"

if [ "$BUILD_AARCH64" = true ]; then
    if [[ -z ${ANDROID_NDK_ROOT} ]]; then
        echo "Please export ANDROID_NDK_ROOT=/path/to/android_ndk"
        exit -1
    fi
    BUILD_ROOT=$PRJ_ROOT/$CMAKE_AARCH64
    if [ "$CLEAN" = true ]; then
        rm -rf $BUILD_ROOT && mkdir $BUILD_ROOT
    fi

    cd $BUILD_ROOT
    # If we build debug type, we need to change flatcc to flatcc_d
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DEXECUTORCH_BUILD_SDK=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
        -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI='arm64-v8a' \
        -DANDROID_NATIVE_API_LEVEL=23 \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -B$BUILD_ROOT

    cmake --build $BUILD_ROOT -j16 --target install

    EXAMPLE_ROOT=examples/qualcomm
    CMAKE_PREFIX_PATH="${BUILD_ROOT}/lib/cmake/ExecuTorch;${BUILD_ROOT}/third-party/gflags;"

    cmake $PRJ_ROOT/$EXAMPLE_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI='arm64-v8a' \
        -DANDROID_NATIVE_API_LEVEL=23 \
        -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -B$EXAMPLE_ROOT

    cmake --build $EXAMPLE_ROOT -j16
fi

if [ "$BUILD_X86_64" = true ]; then
    # Build python interface
    BUILD_ROOT=$PRJ_ROOT/$CMAKE_X86_64
    if [ "$CLEAN" = true ]; then
        rm -rf $BUILD_ROOT && mkdir $BUILD_ROOT
    fi
    cd $BUILD_ROOT
    cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -DBUCK2=$BUCK2 \
        -S $PRJ_ROOT \
        -B $BUILD_ROOT \

    cmake \
    --build $BUILD_ROOT \
    -t "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j16

    rm -f $PRJ_ROOT/backends/qualcomm/python/*
    cp -fv $BUILD_ROOT/backends/qualcomm/Py* "$PRJ_ROOT/backends/qualcomm/python"
fi

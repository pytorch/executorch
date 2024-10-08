# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -e
set -o xtrace

if [[ -z ${QNN_SDK_ROOT} ]]; then
    echo "Please export QNN_SDK_ROOT=/path/to/qnn_sdk"
    exit -1
fi



usage() {
  echo "Usage: Build the aarch64 version of executor runner or the python interface of Qnn Manager"
  echo "First, you need to set the environment variable for QNN_SDK_ROOT"
  echo ", and if you want to build the aarch64 version of executor runner"
  echo ", you need to export ANDROID_NDK_ROOT=/path/to/android_ndkXX"
  echo "e.g.: executorch$ ./backends/qualcomm/scripts/build.sh --skip_x86_64"
  exit 1
}


[ "$1" = -h ] && usage

BUILD_X86_64="true"
CMAKE_X86_64="build-x86"
BUILD_AARCH64="true"
CMAKE_AARCH64="build-android"
CLEAN="true"
BUILD_TYPE="Debug"
BUILD_JOB_NUMBER="16"

if [ -z PYTHON_EXECUTABLE ]; then
  PYTHON_EXECUTABLE="python3"
fi

if [ -z BUCK2 ]; then
  BUCK2="buck2"
fi

long_options=skip_x86_64,skip_aarch64,no_clean,release,job_number:

parsed_args=$(getopt -a --options '' --longoptions $long_options --name "$0" -- "$@")
eval set -- "$parsed_args"


while true ; do
    case "$1" in
        --skip_x86_64) BUILD_X86_64="false"; shift;;
        --skip_aarch64) BUILD_AARCH64="false"; shift;;
        --no_clean) CLEAN="false"; shift;;
        --release) BUILD_TYPE="Release"; shift;;
        --job_number) BUILD_JOB_NUMBER="$2"; shift 2;;
        --) shift; break;;
    esac
done

PRJ_ROOT="$( cd "$(dirname "$0")/../../.." ; pwd -P)"

if [ "$BUILD_AARCH64" = true ]; then
    if [[ -z ${ANDROID_NDK_ROOT} ]]; then
        echo "Please export ANDROID_NDK_ROOT=/path/to/android_ndkXX"
        exit -1
    fi

    BUILD_ROOT=$PRJ_ROOT/$CMAKE_AARCH64
    if [ "$CLEAN" = true ]; then
        rm -rf $BUILD_ROOT && mkdir $BUILD_ROOT
    else
        # Force rebuild flatccrt for the correct platform
        cd $BUILD_ROOT/devtools && make clean
    fi

    cd $BUILD_ROOT
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
        -DQNN_SDK_ROOT=$QNN_SDK_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI='arm64-v8a' \
        -DANDROID_NATIVE_API_LEVEL=23 \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -B$BUILD_ROOT

    cmake --build $BUILD_ROOT -j$BUILD_JOB_NUMBER --target install

    EXAMPLE_ROOT=examples/qualcomm
    CMAKE_PREFIX_PATH="${BUILD_ROOT}/lib/cmake/ExecuTorch;${BUILD_ROOT}/third-party/gflags;"

    cmake $PRJ_ROOT/$EXAMPLE_ROOT \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DANDROID_ABI='arm64-v8a' \
        -DANDROID_NATIVE_API_LEVEL=23 \
        -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
        -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -B$EXAMPLE_ROOT

    cmake --build $EXAMPLE_ROOT -j$BUILD_JOB_NUMBER
fi

if [ "$BUILD_X86_64" = true ]; then
    BUILD_ROOT=$PRJ_ROOT/$CMAKE_X86_64
    if [ "$CLEAN" = true ]; then
        rm -rf $BUILD_ROOT && mkdir $BUILD_ROOT
    else
        # Force rebuild flatccrt for the correct platform
        cd $BUILD_ROOT/devtools && make clean
    fi

    cd $BUILD_ROOT
    cmake \
        -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
        -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT \
        -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
        -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
        -S $PRJ_ROOT \
        -B $BUILD_ROOT \

    cmake --build $BUILD_ROOT -j$BUILD_JOB_NUMBER --target install

    rm -f $PRJ_ROOT/backends/qualcomm/python/*
    cp -fv $BUILD_ROOT/backends/qualcomm/Py* "$PRJ_ROOT/backends/qualcomm/python"
    cp -fv "$PRJ_ROOT/schema/program.fbs" "$PRJ_ROOT/exir/_serialize/program.fbs"
    cp -fv "$PRJ_ROOT/schema/scalar_type.fbs" "$PRJ_ROOT/exir/_serialize/scalar_type.fbs"

   EXAMPLE_ROOT=examples/qualcomm
   CMAKE_PREFIX_PATH="${BUILD_ROOT}/lib/cmake/ExecuTorch;${BUILD_ROOT}/third-party/gflags;"

   cmake $PRJ_ROOT/$EXAMPLE_ROOT \
       -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
       -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
       -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
       -DPYTHON_EXECUTABLE=$PYTHON_EXECUTABLE \
       -B$EXAMPLE_ROOT

   cmake --build $EXAMPLE_ROOT -j$BUILD_JOB_NUMBER
fi

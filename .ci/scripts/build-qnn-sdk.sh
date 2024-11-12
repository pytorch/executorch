#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux
set -o xtrace

build_qnn_backend() {
  echo "Start building qnn backend."
  export ANDROID_NDK_ROOT=/opt/ndk
  export QNN_SDK_ROOT=/tmp/qnn/2.25.0.240728
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  bash backends/qualcomm/scripts/build.sh --skip_aarch64 --job_number 2 --release
}

set_up_aot() {
  cd $EXECUTORCH_ROOT
  if [ ! -d "cmake-out" ]; then
      mkdir cmake-out
  fi
  pushd cmake-out
  cmake .. \
      -DCMAKE_INSTALL_PREFIX=$PWD \
      -DEXECUTORCH_BUILD_QNN=ON \
      -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
      -DEXECUTORCH_BUILD_DEVTOOLS=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
      -DPYTHON_EXECUTABLE=python3 \
      -DEXECUTORCH_SEPARATE_FLATCC_HOST_PROJECT=OFF
  cmake --build $PWD --target "PyQnnManagerAdaptor" "PyQnnWrapperAdaptor" -j$(nproc)
  # install Python APIs to correct import path
  # The filename might vary depending on your Python and host version.
  cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
  cp -f backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
  popd

  # Workaround for fbs files in exir/_serialize
  cp schema/program.fbs exir/_serialize/program.fbs
  cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
}

build_qnn_backend
set_up_aot

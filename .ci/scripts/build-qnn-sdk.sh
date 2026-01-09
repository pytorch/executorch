#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2025 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux
set -o xtrace

build_qnn_backend() {
  echo "Start building qnn backend."
  # Source QNN configuration
  source "$(dirname "${BASH_SOURCE[0]}")/../../backends/qualcomm/scripts/install_qnn_sdk.sh"
  setup_android_ndk
  install_qnn
  export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

  parallelism=$(( $(nproc) - 1 ))
  bash backends/qualcomm/scripts/build.sh --skip_linux_android --skip_linux_embedded --job_number ${parallelism} --release
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
      -DANDROID_NATIVE_API_LEVEL=30 \
      -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
      -DEXECUTORCH_BUILD_DEVTOOLS=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_EXTENSION_LLM=ON \
      -DEXECUTORCH_BUILD_EXTENSION_EXTENSION_LLM_RUNNER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
      -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
      -DPYTHON_EXECUTABLE=python3
  cmake --build $PWD --target "PyQnnManagerAdaptor" -j$(nproc)
  # install Python APIs to correct import path
  # The filename might vary depending on your Python and host version.
  cp -f backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so $EXECUTORCH_ROOT/backends/qualcomm/python
  popd

  # Workaround for fbs files in exir/_serialize
  cp schema/program.fbs exir/_serialize/program.fbs
  cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
}

build_qnn_backend
set_up_aot

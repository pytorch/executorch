#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

if [[ -z "${BUCK:-}" ]]; then
  BUCK=buck2
fi

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

cmake_install_ggml() {
    cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -Bcmake-out/examples/third-party/llama.cpp \
    examples/third-party/llama.cpp

    cmake --build cmake-out/examples/third-party/llama.cpp -j9 --config Debug --target install
}

cmake_install_executorch() {
    cmake \
    -DBUCK2=BUCK \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_PYBIND=ON \
    -Bcmake-out .

    cmake --build cmake-out -j9 --config Debug --target install
}

cmake_install_custom_op() {
    cmake \
    -DBUCK2=BUCK \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -Bcmake-out/examples/models/llama2/playground \
    examples/models/llama2/playground

    cmake --build cmake-out/examples/models/llama2/playground -j9 --config Debug --target install
}

cmake_install_ggml
cmake_install_executorch
cmake_install_custom_op

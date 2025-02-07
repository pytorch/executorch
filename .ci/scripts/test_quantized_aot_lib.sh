#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

which "${PYTHON_EXECUTABLE}"
# Just set this variable here, it's cheap even if we use buck2
CMAKE_OUTPUT_DIR=cmake-out

build_cmake_quantized_aot_lib() {
  echo "Building quantized aot lib"
  SITE_PACKAGES="$(${PYTHON_EXECUTABLE} -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  CMAKE_PREFIX_PATH="${SITE_PACKAGES}/torch"
  (rm -rf ${CMAKE_OUTPUT_DIR} \
    && mkdir ${CMAKE_OUTPUT_DIR} \
    && cd ${CMAKE_OUTPUT_DIR} \
    && retry cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=ON \
      -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" ..)

  cmake --build ${CMAKE_OUTPUT_DIR} -j4
}

build_cmake_quantized_aot_lib

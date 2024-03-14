#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu
# shellcheck source=/dev/null
cmake_install_ggml() {
    cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -Bcmake-out/examples/third-party/llama.cpp \
    examples/third-party/llama.cpp

    cmake --build cmake-out/examples/third-party/llama.cpp -j9 --config Debug --target install
}

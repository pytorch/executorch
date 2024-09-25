#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Ccache (https://ccache.dev/) is used for compiler cache and could make
# ExecuTorch build faster.
# Since CMake 3.17, simply export CMAKE_C_COMPILER_LAUNCHER and CMAKE_CXX_COMPILER_LAUNCHER
# to ccache path and all invocations will use Ccache automatically.

if ! command -v "ccache"; then
  echo "Ccache not found. Installing from Conda." \
       "You can also install from apt, dnf, brew, or official website for latest version"
  conda install -y ccache
fi
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

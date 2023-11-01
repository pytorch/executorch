# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Config defining how CMake should find ExecuTorch package. CMake will search
# for this file and find ExecuTorch package if it is installed.
# Typical usage is:
#
# find_package(executorch REQUIRED)

cmake_minimum_required(VERSION 3.19)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../..")
add_library(executorch STATIC IMPORTED)
find_library(
    EXECUTORCH_LIBRARY_PATH executorch HINTS "${_root}"
)
set_target_properties(
    executorch PROPERTIES IMPORTED_LOCATION "${EXECUTORCH_LIBRARY_PATH}"
)
target_include_directories(executorch INTERFACE ${_root})

add_library(portable_kernels STATIC IMPORTED)
find_library(
    PORTABLE_KERNELS_PATH portable_kernels HINTS "${_root}"
)
set_target_properties(
    portable_kernels PROPERTIES IMPORTED_LOCATION "${PORTABLE_KERNELS_PATH}"
)
target_include_directories(portable_kernels INTERFACE ${_root})

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Config defining how CMake should find ExecuTorch package. CMake will search
# for this file and find ExecuTorch package if it is installed. Typical usage
# is:
#
# find_package(executorch REQUIRED)
# -------
#
# Finds the ExecuTorch library
#
# This will define the following variables:
#
# EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
# EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
# EXECUTORCH_LIBRARIES    -- Libraries to link against
#
# The actual values for these variables will be different from what
# executorch-config.cmake in executorch pip package gives, but we wanted to keep
# the contract of exposing these CMake variables.

cmake_minimum_required(VERSION 3.19)
include("${CMAKE_CURRENT_LIST_DIR}/Utils.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ExecuTorchTargets.cmake")

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_INCLUDE_DIRS
    ${_root}/include ${_root}/include/executorch/runtime/core/portable_type/c10
    ${_root}/lib
)

# TODO: does ExecuTorchTargets.cmake set EXECUTORCH_FOUDND?

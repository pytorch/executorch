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
#   EXECUTORCH_FOUND        -- True if the system has the ExecuTorch library
#   EXECUTORCH_INCLUDE_DIRS -- The include directories for ExecuTorch
#   EXECUTORCH_LIBRARIES    -- Libraries to link against
#
cmake_minimum_required(VERSION 3.19)

# Find prebuilt _portable_lib.so. This file should be installed under
# <site-packages>/executorch/share/cmake
find_library(_portable_lib_LIBRARY _portable_lib.so PATHS "${CMAKE_CURRENT_LIST_DIR}/../../extension/pybindings/")
set(EXECUTORCH_LIBRARIES)
set(EXECUTORCH_FOUND OFF)
if(_portable_lib_LIBRARY)
  set(EXECUTORCH_FOUND ON)
  message(STATUS "ExecuTorch portable library is found at ${_portable_lib_LIBRARY}")
  list(APPEND EXECUTORCH_LIBRARIES _portable_lib)
  add_library(_portable_lib STATIC IMPORTED)
  set(EXECUTORCH_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../include)
  set_target_properties(_portable_lib PROPERTIES
    IMPORTED_LOCATION "${_portable_lib_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${EXECUTORCH_INCLUDE_DIRS}"
    CXX_STANDARD 17
  )
endif()

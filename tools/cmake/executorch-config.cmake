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

cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/Utils.cmake")
find_package(tokenizers REQUIRED)

set(_root "${CMAKE_CURRENT_LIST_DIR}/../../..")
set(EXECUTORCH_INCLUDE_DIRS
    ${_root}/include ${_root}/include/executorch/runtime/core/portable_type/c10
    ${_root}/lib
)
set(non_exported_lib_list XNNPACK xnnpack-microkernels-prod kleidiai pthreadpool cpuinfo)
foreach(lib ${non_exported_lib_list})
  # Name of the variable which stores result of the find_library search
  set(lib_var "LIB_${lib}")
  find_library(
    ${lib_var} ${lib}
    HINTS "${_root}/lib"
    CMAKE_FIND_ROOT_PATH_BOTH
  )
  if(NOT ${lib_var})
    message("${lib} library is not found.
            If needed rebuild with the proper options in CMakeLists.txt"
    )
  else()
    add_library(${lib} STATIC IMPORTED)
    set_target_properties(${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}")
    target_include_directories(
      ${lib}
      INTERFACE ${EXECUTORCH_INCLUDE_DIRS}
    )
    list(APPEND EXECUTORCH_LIBRARIES ${lib})
  endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/ExecuTorchTargets.cmake")

# TODO: does ExecuTorchTargets.cmake set EXECUTORCH_FOUND?

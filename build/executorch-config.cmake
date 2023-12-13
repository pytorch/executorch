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

set(lib_list etdump bundled_program extension_data_loader flatcc_d)
foreach(lib ${lib_list})
    # Name of the variable which stores result of the find_library search
    set(lib_var "LIB_${lib}")
    find_library(${lib_var} ${lib} HINTS "${_root}")
    if(NOT ${lib_var})
        message("${lib} library is not found.
            If needed rebuild with EXECUTORCH_BUILD_SDK=ON")
    else()
        add_library(${lib} STATIC IMPORTED)
        set_target_properties(
            ${lib} PROPERTIES IMPORTED_LOCATION "${${lib_var}}"
        )
        target_include_directories(${lib} INTERFACE ${_root})
    endif()
endforeach()

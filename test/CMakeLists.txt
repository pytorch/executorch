# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for size_test demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(size_test)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Find prebuilt executorch library

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

# Since extract_sources.py is not returning absolute values, we need to patch
# the source paths.
list(TRANSFORM _size_test__srcs PREPEND "${EXECUTORCH_ROOT}/")

set(EXECUTORCH_BUILD_SIZE_TEST_KERNELS
    "none"
    CACHE
      STRING
      "Which kernel library to link into size_test: none, portable, or optimized"
)
set_property(
  CACHE EXECUTORCH_BUILD_SIZE_TEST_KERNELS PROPERTY STRINGS none portable
                                                    optimized
)

# `set_property STRINGS` is a hint to cmake-gui only; enforce the allowed set
# with IN_LIST so a typo fails loudly at configure time.
set(_valid_size_test_kernels none portable optimized)
if(NOT EXECUTORCH_BUILD_SIZE_TEST_KERNELS IN_LIST _valid_size_test_kernels)
  message(
    FATAL_ERROR
      "EXECUTORCH_BUILD_SIZE_TEST_KERNELS='${EXECUTORCH_BUILD_SIZE_TEST_KERNELS}' "
      "must be one of: ${_valid_size_test_kernels}"
  )
endif()

#
# size_test: minimal binary, optionally linked with a kernel library.
#
add_executable(size_test ${_size_test__srcs})
target_link_libraries(size_test executorch extension_data_loader)
if(EXECUTORCH_BUILD_SIZE_TEST_KERNELS STREQUAL "portable")
  target_link_libraries(size_test portable_ops_lib portable_kernels)
elseif(EXECUTORCH_BUILD_SIZE_TEST_KERNELS STREQUAL "optimized")
  if(NOT EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
    message(FATAL_ERROR "EXECUTORCH_BUILD_SIZE_TEST_KERNELS=optimized requires "
                        "EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON"
    )
  endif()
  target_link_libraries(size_test optimized_native_cpu_ops_lib)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(size_test)
endif()

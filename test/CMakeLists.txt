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

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

# Find prebuilt executorch library
find_package(executorch CONFIG REQUIRED)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)
target_include_directories(executorch INTERFACE ${_common_include_directories})

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/../executorch_srcs.cmake")

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# Since extract_sources.py is not returning absolute values, we need to patch
# the source paths.
list(TRANSFORM _size_test__srcs PREPEND "${EXECUTORCH_ROOT}/")

#
# size_test: minimal binary with no ops and no delegate backend
#
# TODO(larryliu0820): Add EXECUTORCH_BUILD_EXECUTABLES to not build executable
# when we cross compile to ios
add_executable(size_test ${_size_test__srcs})
target_link_libraries(size_test executorch)
if(CMAKE_BUILD_TYPE EQUAL "Release")
  target_link_options(size_test PRIVATE "LINKER:--gc-sections")
endif()

#
# size_test_all_ops: binary with portable ops and no delegate backend
#
add_executable(size_test_all_ops ${_size_test__srcs})
target_link_options_shared_lib(portable_ops_lib)
target_link_libraries(
  size_test_all_ops executorch portable_ops_lib portable_kernels
)
if(CMAKE_BUILD_TYPE EQUAL "Release")
  target_link_options(size_test_all_ops PRIVATE "LINKER:--gc-sections")
endif()

# Print all summary
executorch_print_configuration_summary()

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
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Since extract_sources.py is not returning absolute values, we need to patch
# the source paths. TODO(larryliu0820): Fix this
set(_updated_size_test__srcs)
foreach(_src ${_size_test__srcs})
  list(APPEND _updated_size_test__srcs "${EXECUTORCH_ROOT}/${_src}")
endforeach()

#
# size_test: minimal binary with no ops and no delegate backend
#
# TODO(larryliu0820): Add EXECUTORCH_BUILD_EXECUTABLES to not build executable
# when we cross compile to ios
add_executable(size_test ${_updated_size_test__srcs})
target_link_libraries(size_test executorch)
if(CMAKE_BUILD_TYPE EQUAL "Release")
  target_link_options(size_test PRIVATE "LINKER:--gc-sections")
endif()

# Print all summary
executorch_print_configuration_summary()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for runtime components.
#
# ### One-time setup ###
#
# Configure the CMake build system. It's good practice to do this whenever
# cloning or pulling the upstream repo. Once this is done, you don't need to do
# it again until you pull from the upstream repo again.
#
# NOTE: If your `buck2` binary is not on the PATH, you can change this line to
# say something like `-DBUCK2=/tmp/buck2` to point directly to the tool.
#[[
  (rm -rf cmake-out \
    && mkdir cmake-out \
    && cd cmake-out \
    && cmake -DBUCK2=buck2 ..)
]]
#
# ### Build ###
#
# NOTE: The `-j` argument specifies how many jobs/processes to use when
# building, and tends to speed up the build significantly. It's typical to use
# "core count + 1" as the `-j` value.
# ~~~
# cmake --build cmake-out -j9
# ~~~
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
project(executorch)
include(build/Utils.cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
# -Os: Optimize for size -ffunction-sections -fdata-sections: breaks function
# and data into sections so they can be properly gc'd -s: strip symbols
set(CMAKE_CXX_FLAGS_RELEASE
    "-Os -ffunction-sections -fdata-sections -s -fno-exceptions -fno-rtti")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# Option to register custom operator `my_ops::mul3` or `my_ops::mul4` or no
# custom ops at all. Custom ops are defined in
# `examples/custom_ops/custom_ops_1.py` and
# `examples/custom_ops/custom_ops_2.cpp`.
option(
  REGISTER_EXAMPLE_CUSTOM_OP
  "Register whether custom op 1 (my_ops::mul3) or custom op 2 (my_ops::mul4) \
  or no custom op at all."
  OFF)

# Option to register quantized ops with quantized kernels. See
# kernels/quantized/CMakeLists.txt
option(REGISTER_QUANTIZED_OPS
       "Register quantized ops defined in kernels/quantized/" OFF)

option(BUILD_SELECTIVE_BUILD_TEST
       "Whether to build binary for demo selective build" OFF)

if(BUILD_SELECTIVE_BUILD_TEST)
  option(SELECT_ALL_OPS
         "Whether to register all ops defined in portable kernel library." OFF)

  # Option to register op list
  option(SELECT_OPS_LIST "Register the following list of ops" OFF)

  # Option to register ops from yaml file
  option(SELECT_OPS_YAML "Register all the ops from a given yaml file" OFF)
endif()

# Build xnn_executor_runner which depends on XNNPACK
option(EXECUTORCH_BUILD_XNNPACK
       "Build xnn_executor_runner which depends on XNNPACK" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT BUCK2)
  set(BUCK2 buck2)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# TODO(dbort): Fix these warnings and remove this flag.
set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${CMAKE_CURRENT_SOURCE_DIR}/..)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#

if(NOT EXECUTORCH_SRCS_FILE)
  # A file wasn't provided. Run a script to extract the source lists from the
  # buck2 build system and write them to a file we can include.
  #
  # NOTE: This will only happen once during cmake setup, so it will not re-run
  # if the buck2 targets change.
  message(STATUS "executorch: Generating source lists")
  set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/executorch_srcs.cmake")
  extract_sources(${EXECUTORCH_SRCS_FILE})
endif()

# This file defines the `_<target>__srcs` variables used below.
message(STATUS "executorch: Using sources file ${EXECUTORCH_SRCS_FILE}")
include(${EXECUTORCH_SRCS_FILE})

#
# flatc: Flatbuffer commandline tool to generate .h files from .fbs files
#

if(NOT FLATC_EXECUTABLE)
  set(FLATC_EXECUTABLE flatc)
  option(FLATBUFFERS_BUILD_FLATC "" ON)
  option(FLATBUFFERS_BUILD_FLATHASH "" OFF)
  option(FLATBUFFERS_BUILD_FLATLIB "" OFF)
  option(FLATBUFFERS_BUILD_TESTS "" OFF)
  option(FLATBUFFERS_INSTALL "" OFF)
  add_subdirectory(third-party/flatbuffers)
endif()

#
# gflags: Commandline flag libgrary
#

add_subdirectory(third-party/gflags)

#
# program_schema: Generated .h files from schema/*.fbs inputs
#
add_subdirectory(schema)

#
# executorch: Core runtime library
#
# Only contains primitive operators; does not contain portable kernels or other
# full operators. Does not contain any backends.
#

add_library(executorch ${_executorch__srcs})
target_link_libraries(executorch PRIVATE program_schema)
target_link_libraries(executorch PRIVATE dl) # For dladdr()
target_include_directories(executorch PUBLIC ${_common_include_directories})
target_compile_options(executorch PUBLIC ${_common_compile_options})
if(MAX_KERNEL_NUM)
  target_compile_definitions(executorch
                             PRIVATE MAX_KERNEL_NUM=${MAX_KERNEL_NUM})
endif()

#
# portable_ops_lib: A library to register core ATen ops using portable kernels,
# see kernels/portable/CMakeLists.txt.
#
# Real integrations should supply their own YAML file that only lists the
# operators necessary for the models that will run.
#
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/portable)
set(_libs executorch portable_ops_lib gflags)

# Generate custom_ops_lib based on REGISTER_EXAMPLE_CUSTOM_OP
if(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 1 OR REGISTER_EXAMPLE_CUSTOM_OP EQUAL 2)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/custom_ops)
  list(APPEND _libs custom_ops_lib)
endif()

# Generate lib to register quantized ops
if(REGISTER_QUANTIZED_OPS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantized)
  list(APPEND _libs quantized_ops_lib)
endif()

# ios can only build library but not binary
if(NOT CMAKE_TOOLCHAIN_FILE MATCHES ".*ios\.toolchain\.cmake$")
  add_executable(executor_runner ${_executor_runner__srcs})
  if(CMAKE_BUILD_TYPE EQUAL "RELEASE")
    target_link_options(executor_runner PRIVATE "LINKER:--gc-sections")
  endif()
  target_link_libraries(executor_runner ${_libs})
  target_compile_options(executor_runner PUBLIC ${_common_compile_options})
endif()

# Add XNNPACK subdirectory
if(EXECUTORCH_BUILD_XNNPACK)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/backends/xnnpack)
endif()

# Add selective build subdirectory
if(BUILD_SELECTIVE_BUILD_TEST)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/selective_build)
endif()
# Print all summary
executorch_print_configuration_summary()

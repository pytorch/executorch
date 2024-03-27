# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Please this file formatted by running:
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)
project(aot_util)
include(../../build/Utils.cmake)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(NOT EXECUTORCH_ROOT)
    set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../")
endif()

if(NOT BUCK2)
  set(BUCK2 buck2)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

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


# Ahead-of-time (AOT) utility library. Contains native code used by the
# AOT lowering and delegation logic. Note that this library should build
# independently of the runtime code, and as such, should not have
# dependencies on runtime targets.
find_package(Torch CONFIG REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python
               PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Override compiler flags set in upper scope when included from the top-level
# CMakeLists. ExecuTorch builds with -fno-exceptions and -fno-rtti, but we 
# need these for ATen.
unset(CMAKE_CXX_FLAGS_RELEASE)

list(TRANSFORM _extension_aot_util__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(aot_util ${_extension_aot_util__srcs})
target_include_directories(aot_util PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(aot_util torch)

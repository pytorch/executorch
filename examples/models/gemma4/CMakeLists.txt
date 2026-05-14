# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for gemma4 e2e runner.
#
cmake_minimum_required(VERSION 3.24)
project(gemma4)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Need this for gflags for some reason
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# Find `executorch` libraries, same as for gflags
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)
set(_srcs e2e_runner.cpp)

list(
  APPEND
  link_libraries
  optimized_native_cpu_ops_lib
  quantized_ops_lib
  custom_ops
  cpublas
  eigen_blas
)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
executorch_target_link_options_shared_lib(quantized_ops_lib)
executorch_target_link_options_shared_lib(custom_ops)

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add the required ExecuTorch extensions
list(APPEND link_libraries extension_module extension_data_loader
     extension_tensor
)

# Add tokenizers
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(gemma4_e2e_runner ${_srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(gemma4_e2e_runner)
  if(NOT APPLE)
    target_link_options(gemma4_e2e_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(
  gemma4_e2e_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(gemma4_e2e_runner PUBLIC ${link_libraries})
target_compile_options(gemma4_e2e_runner PUBLIC ${_common_compile_options})

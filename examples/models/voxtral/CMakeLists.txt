# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for voxtral runner.
#
cmake_minimum_required(VERSION 3.24)
project(voxtral)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  set(CMAKE_TOOLCHAIN_IOS ON)
else()
  set(CMAKE_TOOLCHAIN_IOS OFF)
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
set(_srcs multimodal.cpp)

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

# Add LLM runner and extension module
if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add the required ExecuTorch extensions for multimodal LLM runner
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Add tokenizers
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(voxtral_runner ${_srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(voxtral_runner)
  if(NOT APPLE)
    target_link_options(voxtral_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(voxtral_runner PUBLIC ${_common_include_directories})
target_link_libraries(voxtral_runner PUBLIC ${link_libraries})
target_compile_options(voxtral_runner PUBLIC ${_common_compile_options})

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(silero_vad_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Need this for gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# Find executorch libraries
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

set(link_libraries executorch gflags)

# Common ops for all builds
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  get_target_property(_xnnpack_imported xnnpack_backend IMPORTED)
  if(NOT _xnnpack_imported)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add the required ExecuTorch extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

add_executable(silero_vad_runner main.cpp silero_vad_runner.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(silero_vad_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(silero_vad_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(
  silero_vad_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(silero_vad_runner PUBLIC ${link_libraries})
target_compile_options(silero_vad_runner PUBLIC ${_common_compile_options})

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(qwen3_tts_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(_link_libraries executorch gflags)

# Common ops for all builds.
list(APPEND _link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# CPU path can require quantized/custom ops when XNNPACK delegates are present.
if(NOT EXECUTORCH_BUILD_CUDA)
  list(APPEND _link_libraries quantized_ops_lib custom_ops)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  executorch_target_link_options_shared_lib(custom_ops)
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND _link_libraries ${xnnpack_backend_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

# Base extensions needed for module loading + tensors.
list(
  APPEND
  _link_libraries
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

if(ANDROID)
  list(APPEND _link_libraries log)
endif()

if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND _link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

add_executable(qwen3_tts_runner main.cpp qwen3_tts_runner.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(qwen3_tts_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(qwen3_tts_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(qwen3_tts_runner PUBLIC ${_common_include_directories})
target_link_libraries(qwen3_tts_runner PUBLIC ${_link_libraries})
target_compile_options(qwen3_tts_runner PUBLIC ${_common_compile_options})

if(MSVC AND EXECUTORCH_BUILD_CUDA)
  add_custom_command(
    TARGET qwen3_tts_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
            $<TARGET_FILE_DIR:qwen3_tts_runner>
    COMMENT "Copying aoti_cuda_shims.dll to qwen3_tts_runner directory"
  )
endif()

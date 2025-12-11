# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.29)
project(whisper_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
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

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(_link_libraries executorch gflags)
set(_srcs multimodal.cpp)

# Common ops for all builds
list(APPEND _link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# CPU-only builds need quantized and custom ops
if(NOT EXECUTORCH_BUILD_CUDA AND MSVC)
  list(APPEND _link_libraries quantized_ops_lib custom_ops)
  executorch_target_link_options_shared_lib(quantized_ops_lib)
  executorch_target_link_options_shared_lib(custom_ops)
else()
  list(APPEND _link_libraries quantized_ops_lib custom_ops)
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

# Add LLM runner and extension module
if(NOT TARGET extension_asr_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_ASR_RUNNER enabled."
  )
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND _link_libraries log)
endif()

# Add the required ExecuTorch extensions for multimodal LLM runner
list(
  APPEND
  _link_libraries
  extension_asr_runner
  extension_llm_runner # Needed for load_tokenizer()
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Link CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND _link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    # On non-MSVC, use shared lib options
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

if(EXECUTORCH_BUILD_METAL)
  list(APPEND _link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
endif()

# Add tokenizers
list(APPEND _link_libraries tokenizers::tokenizers)

add_executable(whisper_runner main.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(whisper_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(whisper_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(whisper_runner PUBLIC ${_common_include_directories})
target_link_libraries(whisper_runner PUBLIC ${_link_libraries})
target_compile_options(whisper_runner PUBLIC ${_common_compile_options})

# On Windows, copy required DLLs to the executable directory
if(MSVC AND EXECUTORCH_BUILD_CUDA)
  add_custom_command(
    TARGET whisper_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
            $<TARGET_FILE_DIR:whisper_runner>
    COMMENT "Copying aoti_cuda_shims.dll to whisper_runner directory"
  )
endif()

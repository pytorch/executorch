# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(gemma4)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)
set(_srcs main.cpp)

list(
  APPEND link_libraries
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

if(NOT TARGET extension_llm_runner)
  message(FATAL_ERROR
    "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled.")
endif()

if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Multimodal support: stb_image for vision input (optional)
option(GEMMA4_ENABLE_VISION "Enable vision/image support (requires stb_image)" OFF)
if(GEMMA4_ENABLE_VISION)
  include(FetchContent)
  FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
  )
  FetchContent_MakeAvailable(stb)
  list(APPEND _common_include_directories ${stb_SOURCE_DIR}
       ${stb_SOURCE_DIR}/deprecated)
  add_definitions(-DGEMMA4_ENABLE_VISION)
endif()

list(
  APPEND link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
endif()

# Tokenizer
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(gemma4_runner ${_srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(gemma4_runner)
  if(NOT APPLE)
    target_link_options(gemma4_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(gemma4_runner PUBLIC ${_common_include_directories})
target_link_libraries(gemma4_runner PUBLIC ${link_libraries})
target_compile_options(gemma4_runner PUBLIC ${_common_compile_options})

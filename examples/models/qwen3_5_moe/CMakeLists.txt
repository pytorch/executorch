# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(qwen3_5_moe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# executorch
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(link_libraries executorch gflags)

# CPU ops
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Extensions
list(APPEND link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# CUDA backend (required)
find_package(CUDAToolkit REQUIRED)
list(APPEND link_libraries aoti_cuda_backend)
if(NOT MSVC)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
endif()

# Tokenizer
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(qwen3_5_moe_runner main.cpp)
target_include_directories(qwen3_5_moe_runner PUBLIC ${_common_include_directories})
target_link_libraries(qwen3_5_moe_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(qwen3_5_moe_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(qwen3_5_moe_runner PRIVATE "LINKER:-s")
  endif()
endif()

# On Windows, copy required DLLs to the executable directory
if(MSVC AND EXECUTORCH_BUILD_CUDA)
  add_custom_command(
    TARGET qwen3_5_moe_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
            $<TARGET_FILE_DIR:qwen3_5_moe_runner>
    COMMENT "Copying aoti_cuda_shims.dll to qwen3_5_moe_runner directory"
  )
endif()

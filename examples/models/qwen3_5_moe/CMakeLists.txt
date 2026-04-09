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

# Extensions
list(APPEND link_libraries extension_module extension_data_loader
     extension_tensor extension_flat_tensor
)

# CUDA backend (required)
find_package(CUDAToolkit REQUIRED)
list(APPEND link_libraries aoti_cuda_backend CUDA::cudart)
executorch_target_link_options_shared_lib(aoti_cuda_backend)

# Tokenizer
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(
  qwen3_5_moe_runner
  main.cpp ${EXECUTORCH_ROOT}/runtime/core/device_allocator.cpp
  ${EXECUTORCH_ROOT}/runtime/core/device_memory_buffer.cpp
  ${EXECUTORCH_ROOT}/backends/cuda/runtime/cuda_allocator.cpp
)
target_include_directories(
  qwen3_5_moe_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(qwen3_5_moe_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(qwen3_5_moe_runner)
  target_link_options(qwen3_5_moe_runner PRIVATE "LINKER:-s")
endif()

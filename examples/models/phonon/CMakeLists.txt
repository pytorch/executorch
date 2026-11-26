# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(phonon_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
if(NOT TARGET tokenizers::tokenizers)
  message(
    FATAL_ERROR
      "Build and install ExecuTorch with EXECUTORCH_BUILD_EXTENSION_LLM=ON"
  )
endif()

add_library(phonon STATIC runner.cpp)
target_include_directories(
  phonon
  PUBLIC ${EXECUTORCH_ROOT}/..
  PRIVATE ${EXECUTORCH_ROOT}/third-party/json/include
)
target_link_libraries(
  phonon
  PUBLIC executorch
         extension_module
         extension_data_loader
         extension_tensor
         extension_flat_tensor
         optimized_native_cpu_ops_lib
         cpublas
         eigen_blas
         tokenizers::tokenizers
)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Kernels for the ops that stay on CPU: the flow LM's KV cache attention
# (llama::sdpa_with_kv_cache) and --quantize quantized ops.
foreach(lib custom_ops quantized_ops_lib)
  if(TARGET ${lib})
    target_link_libraries(phonon PUBLIC ${lib})
    executorch_target_link_options_shared_lib(${lib})
  endif()
endforeach()

if(TARGET xnnpack_backend)
  target_link_libraries(
    phonon PUBLIC xnnpack_backend XNNPACK xnnpack-microkernels-prod
  )
  if(TARGET kleidiai)
    target_link_libraries(phonon PUBLIC kleidiai)
  endif()
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(phonon PUBLIC aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()
if(EXECUTORCH_BUILD_VULKAN)
  if(NOT TARGET vulkan_backend)
    message(
      FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_VULKAN=ON"
    )
  endif()
  executorch_target_link_options_shared_lib(vulkan_backend)
  target_link_libraries(phonon PUBLIC vulkan_backend)
endif()

if(EXECUTORCH_BUILD_WEBGPU)
  # The installed webgpu_backend links dawn::webgpu_dawn; pass -DDawn_DIR.
  find_package(Dawn REQUIRED)
  if(NOT TARGET webgpu_backend)
    message(
      FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_WEBGPU=ON"
    )
  endif()
  executorch_target_link_options_shared_lib(webgpu_backend)
  target_link_libraries(phonon PUBLIC webgpu_backend)
endif()

add_executable(phonon_runner main.cpp)
target_link_libraries(phonon_runner PRIVATE phonon gflags)
if(EXECUTORCH_BUILD_CUDA)
  target_compile_definitions(phonon_runner PRIVATE EXECUTORCH_BUILD_CUDA)
endif()
if(EXECUTORCH_BUILD_MLX)
  if(NOT TARGET mlxdelegate OR NOT TARGET mlx)
    message(
      FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_MLX=ON"
    )
  endif()
  executorch_target_link_options_shared_lib(mlxdelegate)
  target_link_libraries(phonon_runner PRIVATE mlxdelegate mlx)
  executorch_target_copy_mlx_metallib(phonon_runner)
endif()
if(MSVC AND EXECUTORCH_BUILD_CUDA)
  add_custom_command(
    TARGET phonon_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:aoti_cuda_shims> $<TARGET_FILE_DIR:phonon_runner>
  )
endif()

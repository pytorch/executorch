# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(nemotron3_diarization_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
if(NOT TARGET xnnpack_backend)
  message(
    FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_XNNPACK=ON"
  )
endif()
executorch_target_link_options_shared_lib(xnnpack_backend)

add_library(nemotron3_diarization STATIC runner.cpp)
target_include_directories(nemotron3_diarization PUBLIC ${EXECUTORCH_ROOT}/..)
target_link_libraries(
  nemotron3_diarization
  PUBLIC executorch
         extension_module
         extension_data_loader
         extension_tensor
         extension_flat_tensor
         optimized_native_cpu_ops_lib
         cpublas
         eigen_blas
         xnnpack_backend
         XNNPACK
         xnnpack-microkernels-prod
)
if(TARGET kleidiai)
  target_link_libraries(nemotron3_diarization PUBLIC kleidiai)
endif()
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
add_executable(nemotron3_diarization_runner main.cpp)
target_include_directories(
  nemotron3_diarization_runner
  PRIVATE ${EXECUTORCH_ROOT}/third-party/json/include
)
target_link_libraries(
  nemotron3_diarization_runner PRIVATE nemotron3_diarization gflags
)
if(EXECUTORCH_BUILD_MLX)
  if(NOT TARGET mlxdelegate OR NOT TARGET mlx)
    message(
      FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_MLX=ON"
    )
  endif()
  executorch_target_link_options_shared_lib(mlxdelegate)
  target_link_libraries(nemotron3_diarization_runner PRIVATE mlxdelegate mlx)
  target_compile_definitions(
    nemotron3_diarization_runner PRIVATE EXECUTORCH_BUILD_MLX
  )
  executorch_target_copy_mlx_metallib(nemotron3_diarization_runner)
endif()
if(EXECUTORCH_BUILD_VULKAN)
  if(NOT TARGET vulkan_backend)
    message(
      FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_VULKAN=ON"
    )
  endif()
  executorch_target_link_options_shared_lib(vulkan_backend)
  target_link_libraries(nemotron3_diarization PUBLIC vulkan_backend)
endif()

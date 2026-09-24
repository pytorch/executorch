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

# Build MLX here to keep precompiled kernels local to this example.
set(MLX_BUILD_METAL ON)
set(MLX_METAL_JIT OFF)
set(MLX_BUILD_CPU OFF)
set(MLX_BUILD_CUDA OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)
set(MLX_BUILD_PYTHON_STUBS OFF)
set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_GGUF OFF)
set(MLX_BUILD_SAFETENSORS OFF)
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(
  ${EXECUTORCH_ROOT}/backends/mlx/third-party/mlx mlx EXCLUDE_FROM_ALL
)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
if(NOT TARGET mlxdelegate)
  message(
    FATAL_ERROR "Build and install ExecuTorch with EXECUTORCH_BUILD_MLX=ON"
  )
endif()
executorch_target_link_options_shared_lib(mlxdelegate)

add_library(nemotron3_diarization STATIC runner.cpp)
target_include_directories(
  nemotron3_diarization
  PUBLIC ${EXECUTORCH_ROOT}/..
  PRIVATE ${EXECUTORCH_ROOT}/third-party/pocketfft
)
target_link_libraries(
  nemotron3_diarization
  PUBLIC executorch extension_module extension_data_loader extension_tensor
         extension_flat_tensor
)
add_executable(nemotron3_diarization_runner main.cpp)
target_include_directories(
  nemotron3_diarization_runner
  PRIVATE ${EXECUTORCH_ROOT}/third-party/json/include
)
target_link_libraries(
  nemotron3_diarization_runner PRIVATE nemotron3_diarization gflags mlxdelegate
                                       mlx
)
add_custom_command(
  TARGET nemotron3_diarization_runner
  POST_BUILD
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_BINARY_DIR}/mlx/mlx/backend/metal/kernels/mlx.metallib
    $<TARGET_FILE_DIR:nemotron3_diarization_runner>/mlx.metallib
  VERBATIM
)

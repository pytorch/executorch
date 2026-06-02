# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(gemma4_31b)

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

# CPU ops (for the host-side helpers that aren't delegated to CUDA)
list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)

# Extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# Backend: CUDA or MLX (exactly one required)
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
  add_compile_definitions(EXECUTORCH_BUILD_CUDA)
elseif(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
else()
  message(FATAL_ERROR "Set EXECUTORCH_BUILD_CUDA=ON or EXECUTORCH_BUILD_MLX=ON")
endif()

# Tokenizer (HuggingFace tokenizer.json)
list(APPEND link_libraries tokenizers::tokenizers)

# stb_image / stb_image_resize: header-only image loading + bicubic resize.
# Only the CUDA runner has the image (vision) path in this branch, so the stb
# headers are only required for CUDA builds. (MLX vision is added in the
# g4-vision-mlx branch, which removes this gating.) The deprecated/ subdir is
# needed for the (older) stb_image_resize.h API used by image_utils.h.
#
# Resolution order: 1. -DSTB_SOURCE_DIR=/path (or env
# FETCHCONTENT_SOURCE_DIR_STB). 2. ${EXECUTORCH_ROOT}/third-party/stb (vendored
# fallback for proxied/offline environments). 3. FetchContent download from
# github.com/nothings/stb.
if(EXECUTORCH_BUILD_CUDA)
  if(NOT STB_SOURCE_DIR AND EXISTS
                            "${EXECUTORCH_ROOT}/third-party/stb/stb_image.h"
  )
    set(STB_SOURCE_DIR "${EXECUTORCH_ROOT}/third-party/stb")
  endif()
  if(STB_SOURCE_DIR AND EXISTS "${STB_SOURCE_DIR}/stb_image.h")
    message(STATUS "Using local stb headers from: ${STB_SOURCE_DIR}")
    list(APPEND _common_include_directories ${STB_SOURCE_DIR}
         ${STB_SOURCE_DIR}/deprecated
    )
  else()
    include(FetchContent)
    FetchContent_Declare(
      stb
      GIT_REPOSITORY https://github.com/nothings/stb.git
      GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
    )
    FetchContent_MakeAvailable(stb)
    list(APPEND _common_include_directories ${stb_SOURCE_DIR}
         ${stb_SOURCE_DIR}/deprecated
    )
  endif()
endif()

add_executable(gemma4_31b_runner main.cpp)
target_include_directories(
  gemma4_31b_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(gemma4_31b_runner PUBLIC ${link_libraries})

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(gemma4_31b_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(gemma4_31b_runner PRIVATE "LINKER:-s")
  endif()
endif()

if(TARGET mlxdelegate)
  executorch_target_copy_mlx_metallib(gemma4_31b_runner)
endif()

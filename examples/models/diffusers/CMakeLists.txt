# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(diffusion_runner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

set(link_libraries
    executorch
    gflags
    extension_diffusers_runner
    extension_module
    extension_data_loader
    extension_tensor
    extension_flat_tensor
    tokenizers::tokenizers
)

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

if(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
endif()

# stb_image reads the --image input and stb_image_write writes the output PNG.
include(FetchContent)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)
FetchContent_MakeAvailable(stb)

add_executable(diffusion_runner main.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(diffusion_runner)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(diffusion_runner PRIVATE "LINKER:-s")
  endif()
endif()
if(TARGET mlxdelegate)
  executorch_target_copy_mlx_metallib(diffusion_runner)
endif()
target_include_directories(
  diffusion_runner PUBLIC ${_common_include_directories} ${stb_SOURCE_DIR}
)
target_link_libraries(diffusion_runner PUBLIC ${link_libraries})
target_compile_options(diffusion_runner PUBLIC ${_common_compile_options})

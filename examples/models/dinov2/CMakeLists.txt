# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(dinov2_runner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

# Let files say "include <executorch/path/to/header.h>"
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Need this for gflags
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

# Find executorch libraries
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
get_target_property(_executorch_imported executorch IMPORTED)
if(NOT _executorch_imported)
  executorch_target_link_options_shared_lib(executorch)
endif()

set(link_libraries executorch gflags)

# Common ops
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

# Add the required ExecuTorch extensions
list(
  APPEND
  link_libraries
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
)

# stb_image: lightweight library to load and resize images
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

# Link CUDA backend
find_package(CUDAToolkit REQUIRED)
list(APPEND link_libraries aoti_cuda_backend)
if(NOT MSVC)
  executorch_target_link_options_shared_lib(aoti_cuda_backend)
endif()

add_executable(dinov2_runner main.cpp)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(dinov2_runner)
  if(NOT MSVC)
    target_link_options(dinov2_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(
  dinov2_runner PUBLIC ${_common_include_directories}
)
target_link_libraries(dinov2_runner PUBLIC ${link_libraries})
target_compile_options(dinov2_runner PUBLIC ${_common_compile_options})

# On Windows, copy required DLLs to the executable directory
if(MSVC)
  add_custom_command(
    TARGET dinov2_runner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
            $<TARGET_FILE_DIR:dinov2_runner>
    COMMENT "Copying aoti_cuda_shims.dll to dinov2_runner directory"
  )
endif()

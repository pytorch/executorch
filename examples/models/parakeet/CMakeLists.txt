# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
project(parakeet_runner)

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

# Common ops for all builds
if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib cpublas eigen_blas)
  get_target_property(_is_imported optimized_native_cpu_ops_lib IMPORTED)
  if(NOT _is_imported)
    executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
  endif()
endif()

# CPU-only builds need quantized and custom ops
if(NOT EXECUTORCH_BUILD_CUDA)
  if(TARGET quantized_ops_lib)
    list(APPEND link_libraries quantized_ops_lib)
    executorch_target_link_options_shared_lib(quantized_ops_lib)
  endif()
  if(TARGET custom_ops)
    list(APPEND link_libraries custom_ops)
    executorch_target_link_options_shared_lib(custom_ops)
  endif()
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  get_target_property(_xnnpack_imported xnnpack_backend IMPORTED)
  if(NOT _xnnpack_imported)
    executorch_target_link_options_shared_lib(xnnpack_backend)
  endif()
endif()

# Needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add the required ExecuTorch extensions
list(
  APPEND
  link_libraries
  extension_llm_runner
  extension_module
  extension_data_loader
  extension_tensor
  extension_flat_tensor
  tokenizers::tokenizers
)

# Link CUDA backend
if(EXECUTORCH_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND link_libraries aoti_cuda_backend)
  if(NOT MSVC)
    executorch_target_link_options_shared_lib(aoti_cuda_backend)
  endif()
endif()

if(EXECUTORCH_BUILD_METAL)
  list(APPEND link_libraries metal_backend)
  executorch_target_link_options_shared_lib(metal_backend)
endif()

# Link MLX delegate
if(TARGET mlxdelegate)
  list(APPEND link_libraries mlxdelegate mlx)
  executorch_target_link_options_shared_lib(mlxdelegate)
endif()

if(EXECUTORCH_BUILD_VULKAN)
  list(APPEND link_libraries vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
endif()

set(parakeet_shared_sources parakeet_transcriber.cpp timestamp_utils.cpp
                            tokenizer_utils.cpp
)

set(parakeet_common_include_directories
    ${_common_include_directories} ${EXECUTORCH_ROOT}/third-party/json/include
)

add_executable(parakeet_runner main.cpp ${parakeet_shared_sources})
add_executable(
  parakeet_helper parakeet_helper.cpp parakeet_helper_protocol.cpp
                  ${parakeet_shared_sources}
)

foreach(parakeet_target parakeet_runner parakeet_helper)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options_gc_sections(${parakeet_target})
    if(NOT APPLE AND NOT MSVC)
      target_link_options(${parakeet_target} PRIVATE "LINKER:-s")
    endif()
  endif()

  if(TARGET mlxdelegate)
    executorch_target_copy_mlx_metallib(${parakeet_target})
  endif()

  target_include_directories(
    ${parakeet_target} PUBLIC ${parakeet_common_include_directories}
  )
  target_link_libraries(${parakeet_target} PUBLIC ${link_libraries})
  target_compile_options(${parakeet_target} PUBLIC ${_common_compile_options})
endforeach()

# On Windows, copy required DLLs to the executable directory
if(MSVC AND EXECUTORCH_BUILD_CUDA)
  foreach(parakeet_target parakeet_runner parakeet_helper)
    add_custom_command(
      TARGET ${parakeet_target}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:aoti_cuda_shims>
        $<TARGET_FILE_DIR:${parakeet_target}>
      COMMENT "Copying aoti_cuda_shims.dll to ${parakeet_target} directory"
    )
  endforeach()
endif()

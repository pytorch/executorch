# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.24)
if(EXECUTORCH_BUILD_MLX
   AND CMAKE_HOST_APPLE
   AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET
   AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET}
)
  cmake_host_system_information(RESULT _kev_macos_version QUERY OS_RELEASE)
  set(_kev_deployment_target 14.0)
  if(_kev_macos_version VERSION_GREATER_EQUAL 26.2)
    set(_kev_deployment_target 26.2)
  endif()
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      ${_kev_deployment_target}
      CACHE STRING "Minimum macOS version"
  )
endif()
project(kev LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
get_filename_component(EXECUTORCH_ROOT "../.." ABSOLUTE)

option(BUILD_SHARED_LIBS "" OFF)
option(EXECUTORCH_BUILD_EXECUTOR_RUNNER "" OFF)
option(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER "" ON)
option(EXECUTORCH_BUILD_EXTENSION_MODULE "" ON)
option(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR "" ON)
option(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP "" ON)
option(EXECUTORCH_BUILD_EXTENSION_TENSOR "" ON)
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "" ON)
option(EXECUTORCH_BUILD_KERNELS_LLM "" ON)
option(EXECUTORCH_BUILD_XNNPACK "" ON)
option(SUPPORT_REGEX_LOOKAHEAD "" ON)
if(EXECUTORCH_BUILD_MLX)
  set(EXECUTORCH_BUILD_EXTENSION_LLM
      ON
      CACHE BOOL "" FORCE
  )
endif()

add_subdirectory(${EXECUTORCH_ROOT} executorch)
if(NOT TARGET tokenizers::tokenizers)
  add_subdirectory(${EXECUTORCH_ROOT}/extension/llm/tokenizers tokenizers)
endif()

add_library(kev STATIC kev.cpp)
target_link_libraries(
  kev
  PUBLIC extension_module_static extension_tensor tokenizers::tokenizers
  PRIVATE executorch optimized_native_cpu_ops_lib custom_ops
)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
executorch_target_link_options_shared_lib(custom_ops)
if(EXECUTORCH_BUILD_XNNPACK)
  target_link_libraries(kev PRIVATE xnnpack_backend)
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()
if(EXECUTORCH_BUILD_MLX)
  target_link_libraries(kev PRIVATE mlxdelegate)
endif()

add_executable(kev_runner main.cpp)
add_executable(kev_benchmark benchmark.cpp)
foreach(target kev_runner kev_benchmark)
  target_link_libraries(${target} PRIVATE kev)
  if(EXECUTORCH_BUILD_MLX)
    add_custom_command(
      TARGET ${target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MLX_METALLIB_PATH}
              $<TARGET_FILE_DIR:${target}>/mlx.metallib
    )
  endif()
endforeach()

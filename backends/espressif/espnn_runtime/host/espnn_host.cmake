# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

# Host build of the ESP-NN delegate; included from ../CMakeLists.txt outside
# ESP-IDF. Needs the executorch_core target. esp-nn comes from ESPNN_LOCAL_PATH
# or is fetched at ESPNN_GIT_TAG.

include(${CMAKE_CURRENT_LIST_DIR}/../../esp_nn_sources.cmake)
option(ESPNN_HOST_OPTIMIZED "Use esp-nn's generic optimized C kernels" OFF)
option(ESPNN_DELEGATE_PROFILE "Time every delegate operator" OFF)
option(ESPNN_DELEGATE_SCRATCH_IN_ARENA
       "Plan ESP-NN scratch in the activation arena" OFF
)

file(GLOB espnn_kernel_sources "${espnn_source_dir}/src/*/*_ansi.c")
if(ESPNN_HOST_OPTIMIZED)
  file(GLOB espnn_optimized_sources "${espnn_source_dir}/src/*/*_opt.c")
  list(APPEND espnn_kernel_sources ${espnn_optimized_sources})
endif()
add_library(esp_nn_host STATIC ${espnn_kernel_sources})
target_include_directories(
  esp_nn_host
  PUBLIC "${espnn_source_dir}/include"
  PRIVATE "${espnn_source_dir}/src/common"
)
if(ESPNN_HOST_OPTIMIZED)
  target_compile_definitions(esp_nn_host PUBLIC CONFIG_NN_OPTIMIZED=1)
endif()
set_target_properties(esp_nn_host PROPERTIES POSITION_INDEPENDENT_CODE ON)

get_filename_component(
  espnn_executorch_root "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE
)
add_library(
  espnn_backend STATIC
  ${CMAKE_CURRENT_LIST_DIR}/../src/xts3_graph.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/int8_interpreter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/../src/espnn_backend.cpp
)
target_include_directories(
  espnn_backend
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include "${espnn_executorch_root}/.."
          "${espnn_executorch_root}/runtime/core/portable_type/c10"
)
target_compile_definitions(
  espnn_backend PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)
foreach(option ESPNN_DELEGATE_PROFILE ESPNN_DELEGATE_SCRATCH_IN_ARENA)
  if(${option})
    target_compile_definitions(espnn_backend PRIVATE CONFIG_${option}=1)
  endif()
endforeach()
target_link_libraries(espnn_backend PUBLIC executorch_core esp_nn_host)
set_target_properties(espnn_backend PROPERTIES POSITION_INDEPENDENT_CODE ON)
executorch_target_link_options_shared_lib(espnn_backend)

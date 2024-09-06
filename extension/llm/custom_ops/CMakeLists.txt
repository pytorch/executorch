# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Custom op libraries
set(custom_ops_libs pthreadpool)
list(APPEND custom_ops_libs cpuinfo)
list(APPEND custom_ops_libs cpublas)
list(APPEND custom_ops_libs eigen_blas)

list(TRANSFORM _custom_ops__srcs PREPEND "${EXECUTORCH_ROOT}/")

if(NOT EXECUTORCH_BUILD_XNNPACK)
  list(APPEND custom_ops_libs extension_threadpool)
else()
  list(APPEND custom_ops_libs extension_threadpool xnnpack_backend)
endif()

add_library(custom_ops ${_custom_ops__srcs})

target_include_directories(custom_ops PUBLIC "${_common_include_directories}")
target_include_directories(
  custom_ops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../../include"
)
target_link_libraries(
  custom_ops PUBLIC ${custom_ops_libs} executorch_no_prim_ops
)

target_compile_options(
  custom_ops PUBLIC ${_common_compile_options} -DET_USE_THREADPOOL
)

install(TARGETS custom_ops DESTINATION lib)

if(EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT)
  # Add a AOT library
  find_package(Torch CONFIG REQUIRED)
  add_library(
    custom_ops_aot_lib SHARED
    ${_custom_ops__srcs} ${CMAKE_CURRENT_SOURCE_DIR}/op_sdpa_aot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_tile_crop.cpp
  )
  target_include_directories(
    custom_ops_aot_lib PUBLIC "${_common_include_directories}"
  )
  target_include_directories(
    custom_ops_aot_lib PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../../include"
  )
  if(TARGET portable_lib)
    # If we have portable_lib built, custom_ops_aot_lib gives the ability to use
    # the ops in PyTorch and ExecuTorch through pybind
    target_link_libraries(custom_ops_aot_lib PUBLIC portable_lib)
  else()
    # If no portable_lib, custom_ops_aot_lib still gives the ability to use the
    # ops in PyTorch
    target_link_libraries(custom_ops_aot_lib PUBLIC executorch_no_prim_ops)
  endif()

  target_link_libraries(
    custom_ops_aot_lib PUBLIC cpublas torch extension_tensor
                              extension_threadpool
  )
  if(WIN32)
    # There is no direct replacement for libpthread.so on Windows. For the
    # Windows build, link directly against pthreadpool and cpuinfo.
    target_link_libraries(custom_ops_aot_lib PUBLIC pthreadpool cpuinfo)
  endif()
  target_compile_options(
    custom_ops_aot_lib
    PUBLIC -Wno-deprecated-declarations -fPIC -frtti -fexceptions
           ${_common_compile_options} -DET_USE_THREADPOOL
  )

  install(TARGETS custom_ops_aot_lib DESTINATION lib)
endif()

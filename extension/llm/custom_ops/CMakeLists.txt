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

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  list(APPEND _common_compile_options "-march=armv8.2-a+dotprod")
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

set(_common_include_directories
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..,${CMAKE_CURRENT_SOURCE_DIR}/runtime/core/portable_type/c10>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR},${CMAKE_INSTALL_INCLUDEDIR}/executorch/runtime/core/portable_type/c10>
)

list(APPEND _common_include_directories
     $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/third-party/ao>
)

# Custom op libraries
set(custom_ops_libs pthreadpool)
list(APPEND custom_ops_libs cpuinfo)
list(APPEND custom_ops_libs cpublas)
list(APPEND custom_ops_libs eigen_blas)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|armv7)$")
  list(APPEND _custom_ops__srcs
       "extension/llm/custom_ops/spinquant/third-party/FFHT/fht_neon.c"
  )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)")
  list(APPEND _custom_ops__srcs
       "extension/llm/custom_ops/spinquant/third-party/FFHT/fht_avx.c"
  )
else()
  message(
    FATAL_ERROR
      "Unsupported CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}. (If \
32-bit x86, try using fht_avx.c and send a PR if it works!)"
  )
endif()

list(TRANSFORM _custom_ops__srcs PREPEND "${EXECUTORCH_ROOT}/")

if(NOT EXECUTORCH_BUILD_XNNPACK)
  list(APPEND custom_ops_libs extension_threadpool)
else()
  list(APPEND custom_ops_libs extension_threadpool xnnpack_backend)
endif()

add_library(custom_ops ${_custom_ops__srcs})
find_package_torch_headers()
target_include_directories(custom_ops PRIVATE "${_common_include_directories}")
target_include_directories(
  custom_ops
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../../include>
          ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(custom_ops PUBLIC ${custom_ops_libs} executorch_core)

target_compile_options(custom_ops PUBLIC ${_common_compile_options})

install(
  TARGETS custom_ops
  EXPORT ExecuTorchTargets
  DESTINATION lib
)

if(EXECUTORCH_BUILD_KERNELS_LLM_AOT)
  # Add a AOT library
  find_package_torch()
  add_library(
    custom_ops_aot_lib SHARED
    ${_custom_ops__srcs}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_sdpa_aot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_fast_hadamard_transform_aten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_tile_crop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_tile_crop_aot.cpp
  )
  target_include_directories(
    custom_ops_aot_lib PRIVATE "${_common_include_directories}"
  )
  target_include_directories(
    custom_ops_aot_lib
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../../include>
            ${TORCH_INCLUDE_DIRS}
  )
  # TODO: This only works if we install portable_lib.so to
  # <site-packages>/executorch/extension/pybindings/.
  if(APPLE)
    set(RPATH "@loader_path/../../pybindings")
  else()
    set(RPATH "$ORIGIN/../../pybindings")
  endif()
  set_target_properties(custom_ops_aot_lib PROPERTIES INSTALL_RPATH ${RPATH})
  if(TARGET portable_lib)
    # If we have portable_lib built, custom_ops_aot_lib gives the ability to use
    # the ops in PyTorch and ExecuTorch through pybind
    target_link_libraries(custom_ops_aot_lib PUBLIC portable_lib)
  else()
    # If no portable_lib, custom_ops_aot_lib still gives the ability to use the
    # ops in PyTorch
    target_link_libraries(
      custom_ops_aot_lib PUBLIC executorch_core kernels_util_all_deps
    )
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
    custom_ops_aot_lib PUBLIC -Wno-deprecated-declarations -fPIC -frtti
                              -fexceptions ${_common_compile_options}
  )

  install(
    TARGETS custom_ops_aot_lib
    EXPORT ExecuTorchTargets
    LIBRARY DESTINATION executorch/extension/llm/custom_ops
  )
endif()

add_subdirectory(spinquant/third-party/FFHT)
if(BUILD_TESTING)
  add_subdirectory(spinquant/test)
endif()

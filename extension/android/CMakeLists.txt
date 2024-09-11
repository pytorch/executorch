# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

project(executorch_jni)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT ANDROID)
  message(FATAL_ERROR "This directory is for Android build only")
endif()

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

add_subdirectory(
  ${EXECUTORCH_ROOT}/examples/third-party/fbjni
  ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni
)

set(executorch_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED)
target_link_options_shared_lib(executorch)

set(link_libraries)
list(
  APPEND
  link_libraries
  executorch
  extension_data_loader
  extension_module
  extension_runner_util
  extension_tensor
  extension_threadpool
  fbjni
)

if(TARGET optimized_native_cpu_ops_lib)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
endif()
if(TARGET xnnpack_backend)
  target_link_options_shared_lib(xnnpack_backend)
  list(APPEND link_libraries xnnpack_backend XNNPACK pthreadpool cpuinfo)
endif()
if(TARGET vulkan_backend)
  target_link_options_shared_lib(vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
endif()

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/extension/llm/custom_ops
    ${CMAKE_CURRENT_BINARY_DIR}/../../extension/llm/custom_ops
  )
  list(APPEND link_libraries custom_ops)
  target_link_options_shared_lib(custom_ops)
endif()

add_library(executorch_jni SHARED jni/jni_layer.cpp)

if(EXECUTORCH_BUILD_LLAMA_JNI)
  target_sources(executorch_jni PRIVATE jni/jni_layer_llama.cpp)
  list(APPEND link_libraries llama_runner llava_runner)
  target_compile_definitions(executorch_jni PUBLIC EXECUTORCH_BUILD_LLAMA_JNI=1)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/examples/models/llava/runner
    ${CMAKE_CURRENT_BINARY_DIR}/../../examples/models/llava/runner
  )

  add_subdirectory(
    ${EXECUTORCH_ROOT}/examples/models/llama2/runner
    ${CMAKE_CURRENT_BINARY_DIR}/../../examples/models/llama2/runner
  )
endif()

if(TARGET quantized_kernels)
  list(APPEND link_libraries quantized_kernels quantized_ops_lib)
  target_link_options_shared_lib(quantized_ops_lib)
endif()

target_include_directories(
  executorch_jni PRIVATE ${_common_include_directories}
)

target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

target_link_libraries(executorch_jni ${link_libraries})

if(TARGET pthreadpool)
  target_compile_definitions(executorch_jni PRIVATE ET_USE_THREADPOOL=1)
  target_include_directories(
    executorch_jni
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/cpuinfo/include
  )
  target_include_directories(
    executorch_jni
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/pthreadpool/include
  )
endif()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

project(executorch_jni)

if(NOT ANDROID)
  message(FATAL_ERROR "This directory is for Android build only")
endif()

set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
set(_common_compile_options -Wno-deprecated-declarations -fPIC)
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

add_subdirectory(${EXECUTORCH_ROOT}/examples/third-party/fbjni
                 ${CMAKE_CURRENT_BINARY_DIR}/third-party/fbjni)

set(executorch_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED)
target_link_options_shared_lib(executorch)

set(link_libraries)
list(APPEND link_libraries extension_data_loader extension_module executorch
            fbjni)

if(TARGET optimized_native_cpu_ops_lib)
  list(APPEND link_libraries optimized_native_cpu_ops_lib optimized_kernels
              portable_kernels cpublas eigen_blas)
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

add_library(executorch_jni SHARED jni/jni_layer.cpp)
target_link_libraries(executorch_jni ${link_libraries})
target_include_directories(executorch_jni PRIVATE ${_common_include_directories})
target_compile_options(executorch_jni PUBLIC ${_common_compile_options})

if(EXECUTORCH_BUILD_LLAMA_JNI)
  set(LLAMA_RUNNER_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../examples/models/llama2/runner/libllama_runner.a)
  add_library(llama_runner STATIC IMPORTED)
  set_property(TARGET llama_runner PROPERTY IMPORTED_LOCATION ${LLAMA_RUNNER_PATH})

  set(CUSTOM_OPS_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../examples/models/llama2/custom_ops/libcustom_ops_lib.a)
  add_library(custom_ops_lib STATIC IMPORTED)
  set_property(TARGET custom_ops_lib PROPERTY IMPORTED_LOCATION ${CUSTOM_OPS_LIB_PATH})

  set(CUSTOM_OPS_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../examples/models/llama2/custom_ops/libcustom_ops.a)
  add_library(custom_ops STATIC IMPORTED)
  set_property(TARGET custom_ops PROPERTY IMPORTED_LOCATION ${CUSTOM_OPS_PATH})
  target_link_options_shared_lib(custom_ops_lib)

  if(TARGET pthreadpool)
    set(LLAMA_JNI_SRCS jni/jni_layer_llama.cpp ../../backends/xnnpack/threadpool/cpuinfo_utils.cpp)
  else()
    set(LLAMA_JNI_SRCS jni/jni_layer_llama.cpp)
  endif()
  add_library(executorch_llama_jni SHARED ${LLAMA_JNI_SRCS})
  if(TARGET pthreadpool)
    target_compile_definitions(executorch_llama_jni PRIVATE ET_USE_THREADPOOL=1)
    target_include_directories(executorch_llama_jni PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/cpuinfo/include)
    target_include_directories(executorch_llama_jni PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/../../backends/xnnpack/third-party/pthreadpool/include)
  endif()
  target_include_directories(executorch_llama_jni PRIVATE ${_common_include_directories})
  target_link_libraries(executorch_llama_jni ${link_libraries} llama_runner
                        custom_ops custom_ops_lib cpublas eigen_blas)
  target_compile_options(executorch_llama_jni PUBLIC ${_common_compile_options})
endif()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for voxtral runner.
#
cmake_minimum_required(VERSION 3.24) # 3.24 is required for WHOLE_ARCHIVE
project(voxtral)

# Duplicating options as root CMakeLists.txt
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" OFF)

include(CMakeDependentOption)
#
# pthreadpool: build pthreadpool library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)
#
# cpuinfo: build cpuinfo library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF
)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  # Can't set to 11 due to executor_runner.cpp make_unique
endif()

if(CMAKE_TOOLCHAIN_FILE MATCHES ".*(iOS|ios\.toolchain)\.cmake$")
  set(CMAKE_TOOLCHAIN_IOS ON)
else()
  set(CMAKE_TOOLCHAIN_IOS OFF)
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# For some reason android build is not able to find where gflags is and hence
# cannot find corresponding .cmake file
set(gflags_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party/gflags)
find_package(gflags REQUIRED)

#
# voxtral_runner: test binary to run voxtral, with tokenizer and sampler integrated
#

# find `executorch` libraries Same as for gflags
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_CURRENT_BINARY_DIR}/../../..)
find_package(executorch CONFIG REQUIRED FIND_ROOT_PATH_BOTH)
executorch_target_link_options_shared_lib(executorch)

set(LINK_LIBS executorch gflags)
set(link_libraries ${LINK_LIBS})
set(_srcs multimodal.cpp)

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas
  )
  executorch_target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  executorch_target_link_options_shared_lib(portable_ops_lib)
endif()

# quantized_ops_lib: Register quantized op kernels into the runtime
executorch_target_link_options_shared_lib(quantized_ops_lib)
list(APPEND link_libraries quantized_kernels quantized_ops_lib)

# Add custom LLM ops that are needed for llama::update_cache and llama::custom_sdpa
if(TARGET custom_ops)
  executorch_target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
else()
  message(WARNING "custom_ops target not found. LLM custom operators may not be available.")
endif()

set(XNNPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../backends/xnnpack)
# Extra compile option and include dir for pthreadpool
if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND link_libraries extension_threadpool pthreadpool)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/pthreadpool/include
  )
endif()

# Extra sources for cpuinfo
if(EXECUTORCH_BUILD_CPUINFO)
  list(APPEND link_libraries extension_threadpool cpuinfo)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/cpuinfo/include
  )
endif()

# XNNPACK
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK xnnpack-microkernels-prod)
  if(TARGET kleidiai)
    list(APPEND xnnpack_backend_libs kleidiai)
  endif()
  list(APPEND link_libraries ${xnnpack_backend_libs})
  executorch_target_link_options_shared_lib(xnnpack_backend)
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
  executorch_target_link_options_shared_lib(vulkan_backend)
endif()

# Qnn backend
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  executorch_target_link_options_shared_lib(qnn_executorch_backend)
endif()

# MPS backend
if(TARGET mpsdelegate)
  list(
    APPEND
    link_libraries
    mpsdelegate
    "-framework Foundation"
    "-weak_framework MetalPerformanceShaders"
    "-weak_framework MetalPerformanceShadersGraph"
    "-weak_framework Metal"
  )
  executorch_target_link_options_shared_lib(mpsdelegate)
endif()

if(TARGET coremldelegate)
  find_library(SQLITE_LIBRARY sqlite3)
  list(
    APPEND
    link_libraries
    coremldelegate
    sqlite3
    "-framework Foundation"
    "-framework CoreML"
    "-framework Accelerate"
  )
  executorch_target_link_options_shared_lib(coremldelegate)
endif()

# This one is needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

# Add LLM runner and extension module
if(NOT TARGET extension_llm_runner)
  message(
    FATAL_ERROR
      "ExecuTorch must be installed with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER enabled."
  )
endif()

# Add the required ExecutorTorch extensions for multimodal LLM runner
list(APPEND link_libraries 
    extension_llm_runner 
    extension_module 
    extension_data_loader 
    extension_tensor 
    extension_flat_tensor
)

# Add tokenizers
list(APPEND link_libraries tokenizers::tokenizers)

add_executable(voxtral_runner ${_srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(voxtral_runner)
  if(NOT APPLE)
    target_link_options(voxtral_runner PRIVATE "LINKER:-s")
  endif()
endif()

target_include_directories(voxtral_runner PUBLIC ${_common_include_directories})
target_link_libraries(voxtral_runner PUBLIC ${link_libraries})
target_compile_options(voxtral_runner PUBLIC ${_common_compile_options})
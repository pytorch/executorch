# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for llava runner.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.19)
project(llava)

# Duplicating options as root CMakeLists.txt
option(EXECUTORCH_BUILD_KERNELS_OPTIMIZED "Build the optimized kernels" OFF)

# This is a temporary hack to get around Torch dep so we can test this on android
option(LLAVA_RUNNER_NO_TORCH_DUMMY_IMAGE "Hack option to feed dummy image to remove torch.load dep" OFF)

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

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

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

# Avoid torch dep from torch.load()-ing the image.
# This is a temporary hack.
if(LLAVA_RUNNER_NO_TORCH_DUMMY_IMAGE)
  add_definitions(-DLLAVA_NO_TORCH_DUMMY_IMAGE=1)
  message("Buidling the runner without Torch, feeding a dummy image!")
else()
  find_package(Torch CONFIG REQUIRED)
endif()
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

#
# llava_main: test binary to run llava, with tokenizer and sampler integrated
#

# find `executorch` libraries Same as for gflags
set(executorch_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED)
if(CMAKE_TOOLCHAIN_IOS OR ANDROID)
  target_link_options_shared_lib(executorch)
endif()

# custom ops library
if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  add_subdirectory(
    ${EXECUTORCH_ROOT}/extension/llm/custom_ops
    ${CMAKE_CURRENT_BINARY_DIR}/../../../extension/llm/custom_ops
  )
endif()

# llava_runner library
add_subdirectory(runner)

set(LINK_LIBS gflags)
if(NOT LLAVA_RUNNER_NO_TORCH_DUMMY_IMAGE)
  list(APPEND LINK_LIBS torch)
endif()
set(link_libraries ${LINK_LIBS})
set(_srcs main.cpp)

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
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()

# quantized_ops_lib: Register quantized op kernels into the runtime
target_link_options_shared_lib(quantized_ops_lib)
list(APPEND link_libraries quantized_kernels quantized_ops_lib)

if(EXECUTORCH_BUILD_KERNELS_CUSTOM)
  target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
endif()

set(XNNPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../backends/xnnpack)
# Extra compile option and include dir for pthreadpool
if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND _common_compile_options -DET_USE_THREADPOOL)
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
  set(xnnpack_backend_libs xnnpack_backend XNNPACK)
  list(APPEND link_libraries ${xnnpack_backend_libs})
  target_link_options_shared_lib(xnnpack_backend)
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  list(APPEND link_libraries vulkan_backend)
  target_link_options_shared_lib(vulkan_backend)
endif()

# Qnn backend
if(TARGET qnn_executorch_backend)
  list(APPEND link_libraries qnn_executorch_backend)
  target_link_options_shared_lib(qnn_executorch_backend)
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
  target_link_options_shared_lib(mpsdelegate)
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
  target_link_options_shared_lib(coremldelegate)
endif()

# This one is needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

add_executable(llava_main ${_srcs})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(APPLE)
    target_link_options(llava_main PRIVATE "LINKER:-dead_strip,-s")
  else()
    target_link_options(llava_main PRIVATE "LINKER:--gc-sections,-s")
  endif()
endif()

target_include_directories(llava_main PUBLIC ${_common_include_directories})
target_link_libraries(llava_main PUBLIC llava_runner ${link_libraries})
target_compile_options(llava_main PUBLIC ${_common_compile_options})

if(APPLE)
  target_link_options_shared_lib(executorch)
endif()

# Print all summary
executorch_print_configuration_summary()

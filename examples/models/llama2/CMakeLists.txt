# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Simple CMake build system for selective build demo.
#
# ### Editing this file ###
#
# This file should be formatted with
# ~~~
# cmake-format --first-comment-is-literal=True CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.19)
project(llama_runner)

# Duplicating options as root CMakeLists.txt
option(EXECUTORCH_BUILD_OPTIMIZED "Build the optimized kernels" OFF)

option(EXECUTORCH_BUILD_RE2 "Build RE2" OFF)

include(CMakeDependentOption)
#
# pthreadpool: build pthreadpool library. Disable on unsupported platforms
#
cmake_dependent_option(
  EXECUTORCH_BUILD_PTHREADPOOL "Build pthreadpool library." ON
  "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF)
#
# cpuinfo: build cpuinfo library. Disable on unsupported platforms
#
cmake_dependent_option(EXECUTORCH_BUILD_CPUINFO "Build cpuinfo library." ON
                       "NOT EXECUTORCH_BUILD_ARM_BAREMETAL" OFF)

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

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

#
# llama_main: test binary to run llama, with tokenizer and sampler integrated
#

# find `executorch` libraries Same as for gflags
set(executorch_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../../lib/cmake/ExecuTorch)
find_package(executorch CONFIG REQUIRED)
if(CMAKE_TOOLCHAIN_IOS OR ANDROID)
  target_link_options_shared_lib(executorch)
endif()

# custom ops library
if(EXECUTORCH_BUILD_CUSTOM)
  add_subdirectory(custom_ops)
endif()

# llama_runner library
add_subdirectory(runner)
if(EXECUTORCH_BUILD_RE2)
  # find RE2 for tokenizer
  set(ABSL_ENABLE_INSTALL ON)
  set(_pic_flag
    ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/abseil-cpp)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third-party/re2)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})
  target_link_libraries(llama_runner PUBLIC re2::re2)
endif()

set(link_libraries gflags)
set(_srcs main.cpp)

if(EXECUTORCH_BUILD_OPTIMIZED)
  list(
    APPEND
    link_libraries
    optimized_native_cpu_ops_lib
    optimized_kernels
    portable_kernels
    cpublas
    eigen_blas)
  target_link_options_shared_lib(optimized_native_cpu_ops_lib)
else()
  list(APPEND link_libraries portable_ops_lib portable_kernels)
  target_link_options_shared_lib(portable_ops_lib)
endif()

if(EXECUTORCH_BUILD_CUSTOM)
  target_link_options_shared_lib(custom_ops)
  list(APPEND link_libraries custom_ops)
endif()

set(XNNPACK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../backends/xnnpack)
# Extra compile option and include dir for pthreadpool
if(EXECUTORCH_BUILD_PTHREADPOOL)
  list(APPEND _common_compile_options -DET_USE_THREADPOOL)
  list(APPEND link_libraries pthreadpool)
  # These 2 source files are included in xnnpack_backend
  if(NOT TARGET xnnpack_backend)
    list(APPEND _srcs ${XNNPACK_ROOT}/threadpool/threadpool.cpp
         ${XNNPACK_ROOT}/threadpool/threadpool_guard.cpp)
  endif()
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/pthreadpool/include)
endif()

# Extra sources for cpuinfo
if(EXECUTORCH_BUILD_CPUINFO)
  list(APPEND link_libraries cpuinfo)
  list(APPEND _srcs ${XNNPACK_ROOT}/threadpool/cpuinfo_utils.cpp)
  list(APPEND _common_include_directories
       ${XNNPACK_ROOT}/third-party/cpuinfo/include)
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

# This one is needed for cpuinfo where it uses android specific log lib
if(ANDROID)
  list(APPEND link_libraries log)
endif()

add_executable(llama_main ${_srcs})
if(CMAKE_BUILD_TYPE EQUAL "RELEASE")
  target_link_options(llama_main PRIVATE "LINKER:--gc-sections")
endif()

target_include_directories(llama_main PUBLIC ${_common_include_directories})
target_link_libraries(llama_main PUBLIC llama_runner ${link_libraries})
target_compile_options(llama_main PUBLIC ${_common_compile_options})

if(APPLE)
  target_link_options_shared_lib(executorch)
endif()

# Print all summary
executorch_print_configuration_summary()

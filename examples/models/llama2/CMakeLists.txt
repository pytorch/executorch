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

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  # Can't set to 11 due to executor_runner.cpp make_unique
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(gflags REQUIRED PATHS
             ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party)

#
# llama_main: test binary to run llama, with tokenizer and sampler integrated
#
add_executable(llama_main main.cpp)
if(CMAKE_BUILD_TYPE EQUAL "RELEASE")
  target_link_options(llama_main PRIVATE "LINKER:--gc-sections")
endif()

# find `executorch` libraries
find_package(executorch CONFIG REQUIRED)

# llama_runner library
add_subdirectory(runner)


target_link_libraries(llama_main PUBLIC gflags llama_runner
                      portable_ops_lib)
target_link_options(
  llama_main PUBLIC "SHELL:LINKER:--whole-archive \
                    $<TARGET_FILE:portable_ops_lib> \
                    LINKER:--no-whole-archive")

# XNNPACK pthreadpool cpuinfo
if(TARGET xnnpack_backend)
  set(xnnpack_backend_libs xnnpack_backend XNNPACK pthreadpool cpuinfo)
  target_link_libraries(llama_main PUBLIC ${xnnpack_backend_libs})
  target_link_options(
    llama_main PUBLIC "SHELL:LINKER:--whole-archive \
                      $<TARGET_FILE:xnnpack_backend> \
                      LINKER:--no-whole-archive")
endif()

# Vulkan backend
if(TARGET vulkan_backend)
  target_link_libraries(llama_main PUBLIC vulkan_backend)
  target_link_options(
    llama_main PUBLIC "SHELL:LINKER:--whole-archive \
                      $<TARGET_FILE:vulkan_backend> \
                      LINKER:--no-whole-archive")
endif()

target_compile_options(llama_main PUBLIC ${_common_compile_options})

# Print all summary
executorch_print_configuration_summary()

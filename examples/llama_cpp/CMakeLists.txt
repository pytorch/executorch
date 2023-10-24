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
project(LlamaCppExample)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(Llama REQUIRED)
find_package(ExecuTorch REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)

target_include_directories(executorch INTERFACE ${_common_include_directories})

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(
  EXECUTORCH_SRCS_FILE
  "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)
if(NOT EXISTS "${EXECUTORCH_SRCS_FILE}")
  # A file wasn't generated. Run a script to extract the source lists from the
  # buck2 build system and write them to a file we can include.
  #
  # NOTE: This will only happen once during cmake setup, so it will not re-run
  # if the buck2 targets change.
  message(STATUS "executorch: Generating source lists")
  set(EXECUTORCH_SRCS_FILE "${CMAKE_CURRENT_BINARY_DIR}/executorch_srcs.cmake")
  extract_sources(${EXECUTORCH_SRCS_FILE})
endif()

# This file defines the `_<target>__srcs` variables used below.
message(STATUS "executorch: Using sources file ${EXECUTORCH_SRCS_FILE}")
include(${EXECUTORCH_SRCS_FILE})

set(_custom_ops_yaml ${EXECUTORCH_ROOT}/examples/llama_cpp/custom_ops.yaml)
set(_ops_yaml ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml)

set(kernel_sources ${EXECUTORCH_ROOT}/examples/llama_cpp/op_mm.cpp)
#
# custom_kernels: C++ kernel implementations of custom ops
#
add_library(custom_kernels ${kernel_sources})
target_link_libraries(custom_kernels PRIVATE executorch llama)
target_compile_options(custom_kernels PUBLIC ${_common_compile_options})

set(_kernel_lib custom_kernels portable_kernels)

# Select all ops in functions.yaml as well as custom op.
gen_selected_ops("${_ops_yaml}" "ggml::mul_mat.out" "")

#
# kernel_lib: contains both custom_kernels and portable_kernels
#
generate_bindings_for_kernels("${_ops_yaml}" "${_custom_ops_yaml}")
gen_operators_lib("kernel_lib" ${_kernel_lib} executorch)
target_link_libraries(kernel_lib PRIVATE executorch)

list(TRANSFORM _executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

#
# llama_cpp_test: test binary to run llama.cpp kernel ggml_mul_mat
#
add_executable(llama_cpp_test ${_executor_runner__srcs})
if(CMAKE_BUILD_TYPE EQUAL "RELEASE")
  target_link_options(selective_build_test PRIVATE "LINKER:--gc-sections")
endif()
target_link_libraries(llama_cpp_test executorch gflags kernel_lib)
target_compile_options(llama_cpp_test PUBLIC ${_common_compile_options})

# Print all summary
executorch_print_configuration_summary()

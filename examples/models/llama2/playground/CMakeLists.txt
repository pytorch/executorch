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

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(Llama REQUIRED)
find_package(ExecuTorch REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../../../third-party
)

target_include_directories(executorch INTERFACE ${_common_include_directories})


set(kernel_sources ${EXECUTORCH_ROOT}/examples/llama2/playground/op_mm.cpp)
#
# custom_kernels: C++ kernel implementations of custom ops
#
add_library(custom_kernels SHARED ${kernel_sources})
target_link_libraries(custom_kernels PRIVATE executorch ${LLAMA_LIBRARY})
target_compile_options(custom_kernels PUBLIC ${_common_compile_options})
target_include_directories(custom_kernels PRIVATE ${EXECUTORCH_ROOT}/examples/third-party/llama.cpp)


if(EXECUTORCH_BUILD_GTEST)
    #
    # llama_cpp_test: test binary to run llama.cpp kernel ggml_mul_mat
    #
    add_executable(llama_cpp_test PRIVATE ${EXECUTORCH_ROOT}/examples/llama2/playground/test_op_mm.cpp)

    target_link_libraries(llama_cpp_test executorch gflags custom_kernels)
    target_compile_options(llama_cpp_test PUBLIC ${_common_compile_options})
endif()

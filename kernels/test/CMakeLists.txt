# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(kernels_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_kernels portable optimized)

foreach(kernel ${_kernels})
  set(_wrapper_path
      "${CMAKE_CURRENT_BINARY_DIR}/include/${kernel}/executorch/kernels/test/FunctionHeaderWrapper.h"
  )
  add_custom_command(
    OUTPUT "${_wrapper_path}"
    COMMAND mkdir -p include/${kernel}/executorch/kernels/test
    COMMAND echo "#include <${kernel}/Functions.h>" > "${_wrapper_path}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating ${_wrapper_path}"
    VERBATIM
  )
endforeach()

add_custom_target(
  generate_wrapper
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/include/portable/executorch/kernels/test/FunctionHeaderWrapper.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/optimized/executorch/kernels/test/FunctionHeaderWrapper.h"
)

set(_portable_kernels_test_sources op_abs_test.cpp)
set(_optimized_kernels_test_sources op_neg_test.cpp)

et_cxx_test(
  portable_kernels_test SOURCES ${_portable_kernels_test_sources} EXTRA_LIBS
  portable_kernels portable_ops_lib
)
add_dependencies(portable_kernels_test generate_wrapper)
target_include_directories(
  portable_kernels_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/portable"
                                "${CMAKE_INSTALL_PREFIX}/include"
)

et_cxx_test(
  optimized_kernels_test SOURCES ${_optimized_kernels_test_sources} EXTRA_LIBS
  optimized_kernels optimized_ops_lib
)
add_dependencies(optimized_kernels_test generate_wrapper)
# message(FATAL_ERROR "${CMAKE_INSTALL_PREFIX}/include")
target_include_directories(
  optimized_kernels_test PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include/optimized"
                                 "${CMAKE_INSTALL_PREFIX}/include"
)

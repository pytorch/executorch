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
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

#
# select_build_lib: C++ library to register selected ops in custom kernel
# library
#
set(_kernel_lib)
if(EXECUTORCH_SELECT_OPS_YAML)
  set(_custom_ops_yaml
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops.yaml)
  gen_selected_ops("${_custom_ops_yaml}" "" "")
  set(kernel_sources
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops_1_out.cpp
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops_2_out.cpp)
  #
  # custom_kernels: C++ kernel implementations of custom ops
  #
  add_library(custom_kernels ${kernel_sources})
  target_link_libraries(custom_kernels PRIVATE executorch)
  target_compile_options(custom_kernels PUBLIC ${_common_compile_options})

  list(APPEND _kernel_lib custom_kernels)
else()
  list(APPEND _kernel_lib portable_kernels)
endif()

gen_selected_ops(
  "${_custom_ops_yaml}"
  "${EXECUTORCH_SELECT_OPS_LIST}"
  "${EXECUTORCH_SELECT_ALL_OPS}")

generate_bindings_for_kernels(${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
                              "${_custom_ops_yaml}")
gen_operators_lib("select_build_lib" ${_kernel_lib} executorch)

set(_updated__srcs)
foreach(_src ${_executor_runner__srcs})
  list(APPEND _updated__srcs "${EXECUTORCH_ROOT}/${_src}")
endforeach()

#
# selective_build_test: test binary to allow different operator libraries to
# link to
#
add_executable(selective_build_test ${_updated__srcs})
if(CMAKE_BUILD_TYPE EQUAL "RELEASE")
  target_link_options(selective_build_test PRIVATE "LINKER:--gc-sections")
endif()
target_link_libraries(selective_build_test executorch gflags select_build_lib)
target_compile_options(selective_build_test PUBLIC ${_common_compile_options})

# Print all summary
executorch_print_configuration_summary()

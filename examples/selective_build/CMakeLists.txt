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
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#
cmake_minimum_required(VERSION 3.19)
project(selective_build_example)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  # Can't set to 11 due to executor_runner.cpp make_unique
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)

target_include_directories(executorch INTERFACE ${_common_include_directories})

# ------------------------------ OPTIONS BEGIN -------------------------------

# Option to register ops from yaml file
option(EXECUTORCH_SELECT_OPS_YAML "Register all the ops from a given yaml file"
       OFF
)

# Option to register op list
option(EXECUTORCH_SELECT_OPS_LIST "Register a list of ops, separated by comma"
       OFF
)

# Selective build options.
option(EXECUTORCH_SELECT_ALL_OPS
       "Whether to register all ops defined in portable kernel library." OFF
)
# ------------------------------- OPTIONS END --------------------------------

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

#
# select_build_lib: C++ library to register selected ops in custom kernel
# library
#
set(_kernel_lib)
if(EXECUTORCH_SELECT_OPS_YAML)
  set(_custom_ops_yaml
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops.yaml
  )
  set(kernel_sources
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops_1_out.cpp
      ${EXECUTORCH_ROOT}/examples/portable/custom_ops/custom_ops_2_out.cpp
  )
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
  LIB_NAME
  "select_build_lib"
  OPS_SCHEMA_YAML
  "${_custom_ops_yaml}"
  ROOT_OPS
  "${EXECUTORCH_SELECT_OPS_LIST}"
  INCLUDE_ALL_OPS
  "${EXECUTORCH_SELECT_ALL_OPS}"
)

generate_bindings_for_kernels(
  LIB_NAME "select_build_lib" FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml CUSTOM_OPS_YAML
  "${_custom_ops_yaml}"
)

gen_operators_lib(
  LIB_NAME "select_build_lib" KERNEL_LIBS ${_kernel_lib} DEPS executorch
)

list(TRANSFORM _executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

#
# selective_build_test: test binary to allow different operator libraries to
# link to
#
add_executable(selective_build_test ${_executor_runner__srcs})
if(CMAKE_BUILD_TYPE EQUAL "Release")
  target_link_options(selective_build_test PRIVATE "LINKER:--gc-sections")
endif()
target_link_libraries(
  selective_build_test PRIVATE executorch gflags select_build_lib
)
target_link_options_shared_lib(select_build_lib)
target_link_options_shared_lib(executorch)
target_compile_options(selective_build_test PUBLIC ${_common_compile_options})

# Print all summary
executorch_print_configuration_summary()

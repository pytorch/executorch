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

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  # Can't set to 11 due to executor_runner.cpp make_unique
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC
                            -ffunction-sections -fdata-sections
)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../third-party
)

target_include_directories(
  executorch_core INTERFACE ${_common_include_directories}
)

# ------------------------------ OPTIONS BEGIN -------------------------------

# Selective build options.
option(EXECUTORCH_EXAMPLE_SELECT_ALL_OPS
       "Whether to register all ops defined in portable kernel library." OFF
)

option(EXECUTORCH_EXAMPLE_DEFINE_CUSTOM_TARGET
       "Whether to define a custom operator target for advanced use cases. If false, the top-level executorch_kernels target is linked." OFF
)

# Note that the following options are defined by the core framework and
# are also used by this example when defining a custom operator target:
#
# EXECUTORCH_SELECT_OPS_YAML
# EXECUTORCH_SELECT_OPS_LIST
# EXECUTORCH_SELECT_OPS_MODEL
# EXECUTORCH_DTYPE_SELECTIVE_BUILD

# ------------------------------- OPTIONS END --------------------------------

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

if(NOT EXECUTORCH_EXAMPLE_DEFINE_CUSTOM_TARGET)
  # For standard use cases, we can configure the ExecuTorch kernel library
  # build using the EXECUTORCH_SELECT_OPS_* variables. This will reflect in
  # the executorch_kernels target, which includes the configured kernel
  # libraries, including selective build, where supported.

  set(selected_kernel_target executorch_kernels)
else() # EXECUTORCH_EXAMPLE_DEFINE_CUSTOM_TARGET
  # For advanced use cases, we can define a custom operator target.
  # This is useful when using custom operators.
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
    target_link_libraries(custom_kernels PRIVATE executorch_core)
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
    OPS_FROM_MODEL
    "${EXECUTORCH_SELECT_OPS_MODEL}"
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
  )

  generate_bindings_for_kernels(
    LIB_NAME
    "select_build_lib"
    FUNCTIONS_YAML
    ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
    CUSTOM_OPS_YAML
    "${_custom_ops_yaml}"
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
  )

  gen_operators_lib(
    LIB_NAME
    "select_build_lib"
    KERNEL_LIBS
    ${_kernel_lib}
    DEPS
    executorch_core
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_DTYPE_SELECTIVE_BUILD}"
  )

  executorch_target_link_options_shared_lib(select_build_lib)
  set(selected_kernel_target select_build_lib)
endif()

list(TRANSFORM _executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

#
# selective_build_test: test binary to allow different operator libraries to
# link to
#
add_executable(selective_build_test ${_executor_runner__srcs})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_link_options_gc_sections(selective_build_test)
endif()
target_link_libraries(
  selective_build_test PRIVATE executorch_core extension_evalue_util
                               extension_runner_util gflags ${selected_kernel_target}
)
target_compile_options(selective_build_test PUBLIC ${_common_compile_options})

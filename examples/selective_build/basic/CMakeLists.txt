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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
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

add_subdirectory(${EXECUTORCH_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/executorch)

# ------------------------------ OPTIONS BEGIN -------------------------------

# The following options are defined by the core framework and are also used in
# the generated kernel target.
#
# EXECUTORCH_SELECT_OPS_YAML EXECUTORCH_SELECT_OPS_LIST
# EXECUTORCH_SELECT_OPS_MODEL EXECUTORCH_DTYPE_SELECTIVE_BUILD

# ------------------------------- OPTIONS END --------------------------------

#
# The `_<target>_srcs` lists are defined by executorch_load_build_variables.
#
executorch_load_build_variables()

# For most use cases, we can configure the ExecuTorch kernel library build using
# the EXECUTORCH_SELECT_OPS_* variables. This will reflect in the
# executorch_kernels target, which includes the configured kernel libraries,
# including selective build, where supported.

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
  selective_build_test
  PRIVATE executorch_core extension_evalue_util extension_runner_util
          gflags::gflags executorch_kernels
)
target_compile_options(selective_build_test PUBLIC ${_common_compile_options})

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for portable kernels. Please this file formatted by running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~

cmake_minimum_required(VERSION 3.19)
project(arm_example)

# Option to register op list
option(EXECUTORCH_SELECT_OPS_LIST "Register the following list of ops" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED HINTS ${CMAKE_INSTALL_PREFIX})
target_include_directories(executorch INTERFACE ${_common_include_directories})

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops(
  LIB_NAME
  "arm_portable_ops_lib"
  OPS_SCHEMA_YAML
  ""
  ROOT_OPS
  "${EXECUTORCH_SELECT_OPS_LIST}"
  INCLUDE_ALL_OPS
  ""
)
generate_bindings_for_kernels(
  LIB_NAME "arm_portable_ops_lib" FUNCTIONS_YAML
  ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml
)
gen_operators_lib(
  LIB_NAME "arm_portable_ops_lib" KERNEL_LIBS portable_kernels DEPS executorch
)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  target_compile_options(executorch INTERFACE -DET_EVENT_TRACER_ENABLED)
  target_compile_options(portable_ops_lib INTERFACE -DET_EVENT_TRACER_ENABLED)
endif()

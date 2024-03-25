# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)
endif()

# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../executorch_srcs.cmake")

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)


# Custom op libraries
set(custom_ops_libs extension_module)
list(APPEND custom_ops_libs pthreadpool)
list(APPEND custom_ops_libs cpuinfo)

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in optimized.yaml
set(_yaml "${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml")
gen_selected_ops("${_yaml}" "" "")

generate_bindings_for_kernels(
  FUNCTIONS_YAML ${CMAKE_CURRENT_SOURCE_DIR}/custom_ops.yaml)
message("Generated files ${gen_command_sources}")

list(TRANSFORM _custom_ops__srcs PREPEND "${EXECUTORCH_ROOT}/")
add_library(custom_ops ${_custom_ops__srcs})

target_include_directories(custom_ops PUBLIC "${_common_include_directories}")
target_include_directories(custom_ops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../../../include")
target_link_libraries(custom_ops PRIVATE ${custom_ops_libs})

target_compile_options(custom_ops PUBLIC ${_common_compile_options} -DET_USE_THREADPOOL)

# Build a library for _custom_ops_srcs
#
# custom_ops_lib: Register optimized ops kernels into Executorch runtime
gen_operators_lib(
  "custom_ops_lib"
  KERNEL_LIBS custom_ops
  DEPS executorch)

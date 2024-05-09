# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example CMakeLists.txt for registering custom ops into ExecuTorch. In this
# example we have custom ops `my_ops::mul3.out` implemented in C++ in
# `examples/portable/custom_ops/custom_ops_1.cpp`. We also have it registered
# into EXIR in `examples/portable/custom_ops/custom_ops_1.py`. This
# CMakeLists.txt runs a script to generate wrapper code based on the operator
# kernel binding defined in `examples/portable/custom_ops/custom_ops.yaml`. Then
# creates a library that contains both binding wrapper and the implementation
# source file. This library can be linked into ExecuTorch binary
# (`executor_runner` in this example) and it is ready to run models containing
# that custom op.
cmake_minimum_required(VERSION 3.19)
project(custom_ops_example)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations -fPIC)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..)

find_package(executorch CONFIG REQUIRED)
find_package(
  gflags REQUIRED PATHS ${CMAKE_CURRENT_BINARY_DIR}/../../../third-party
)

target_include_directories(executorch INTERFACE ${_common_include_directories})

# ------------------------------ OPTIONS BEGIN -------------------------------
# Option to register custom operator `my_ops::mul3` or `my_ops::mul4` or no
# custom ops at all. Custom ops are defined in
# `examples/portable/custom_ops/custom_ops_1.py` and
# `examples/portable/custom_ops/custom_ops_2.cpp`.
option(
  REGISTER_EXAMPLE_CUSTOM_OP
  "Register whether custom op 1 (my_ops::mul3) or custom op 2 (my_ops::mul4) \
  or no custom op at all." OFF
)
# ------------------------------- OPTIONS END --------------------------------

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime).
if(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 1)
  gen_selected_ops(LIB_NAME "custom_ops_lib" ROOT_OPS "my_ops::mul3.out")
elseif(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 2)
  gen_selected_ops(LIB_NAME "custom_ops_lib" ROOT_OPS "my_ops::mul4.out")
endif()
# Expect gen_selected_ops output file to be selected_operators.yaml
generate_bindings_for_kernels(
  LIB_NAME "custom_ops_lib" CUSTOM_OPS_YAML
  ${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml
)
message("Generated files ${gen_command_sources}")

# Prepare for C++ libraries.

# C++ library to register custom ops into PyTorch.
if(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 2)
  gen_selected_ops(LIB_NAME "custom_ops_aot_lib" ROOT_OPS "my_ops::mul4.out")
  generate_bindings_for_kernels(
    LIB_NAME "custom_ops_aot_lib" CUSTOM_OPS_YAML
    ${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml
  )
  set(custom_ops_kernel_sources
      ${CMAKE_CURRENT_LIST_DIR}/custom_ops_2.cpp # register my_ops::mul4
      ${CMAKE_CURRENT_LIST_DIR}/custom_ops_2_out.cpp # register my_ops::mul4.out
  )
  gen_custom_ops_aot_lib(
    LIB_NAME "custom_ops_aot_lib" KERNEL_SOURCES "${custom_ops_kernel_sources}"
  )
  target_include_directories(
    custom_ops_aot_lib PUBLIC ${_common_include_directories}
  )
endif()

# C++ library to register custom ops into Executorch runtime.
if(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 1)
  set(kernel_sources ${CMAKE_CURRENT_LIST_DIR}/custom_ops_1_out.cpp)
elseif(REGISTER_EXAMPLE_CUSTOM_OP EQUAL 2)
  set(kernel_sources ${CMAKE_CURRENT_LIST_DIR}/custom_ops_2_out.cpp)
endif()

add_library(custom_kernels ${kernel_sources})
target_link_libraries(custom_kernels PRIVATE executorch)
target_compile_options(custom_kernels PUBLIC ${_common_compile_options})

gen_operators_lib(
  LIB_NAME "custom_ops_lib" KERNEL_LIBS custom_kernels DEPS executorch
)

list(TRANSFORM _executor_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")

add_executable(custom_ops_executor_runner ${_executor_runner__srcs})
target_link_libraries(
  custom_ops_executor_runner custom_ops_lib executorch gflags
)
target_compile_options(
  custom_ops_executor_runner PUBLIC ${_common_compile_options}
)

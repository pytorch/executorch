# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example CMakeLists.txt for registering custom ops into Executorch. In this
# example we have custom ops `my_ops::mul3.out` implemented in C++ in
# `examples/custom_ops/custom_ops_1.cpp`. We also have it registered into EXIR
# in `examples/custom_ops/custom_ops_1.py`. This CMakeLists.txt runs a script to
# generate wrapper code based on the operator-kernel binding defined in
# `examples/custom_ops/custom_ops.yaml`. Then creates a library that contains
# both binding wrapper and the implementation source file. This library can be
# linked into Executorch binary (`executor_runner` in this example) and it is
# ready to run models containing that custom op.
cmake_minimum_required(VERSION 3.13)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT PYTHON_EXECUTABLE)
  set(PYTHON_EXECUTABLE python3)
endif()
# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
# Source root directory for pytorch.
if(NOT TORCH_ROOT)
  set(TORCH_ROOT ${EXECUTORCH_ROOT}/third-party/pytorch)
endif()
# Command to generate selected_operators.yaml from custom_ops.yaml.
set(_oplist_yaml ${CMAKE_CURRENT_BINARY_DIR}/selected_operators.yaml)
file(GLOB_RECURSE _codegen_tools_srcs "${EXECUTORCH_ROOT}/codegen/tools/*.py")
file(GLOB_RECURSE _codegen_templates "${EXECUTORCH_ROOT}/codegen/templates/*")
file(GLOB_RECURSE _torchgen_srcs "${TORCH_ROOT}/torchgen/*.py")

set(_gen_oplist_command
    "${PYTHON_EXECUTABLE}" -m codegen.tools.gen_oplist
    --output_path=${_oplist_yaml}
    --ops_schema_yaml_path=${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml)

# Command to codegen C++ wrappers to register custom ops to both PyTorch and
# Executorch runtime.
set(_gen_command
    "${PYTHON_EXECUTABLE}" -m torchgen.gen_executorch
    --source-path=${EXECUTORCH_ROOT}/codegen
    --install-dir=${CMAKE_CURRENT_BINARY_DIR}
    --tags-path=${TORCH_ROOT}/aten/src/ATen/native/tags.yaml
    --aten-yaml-path=${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml
    --op-selection-yaml-path=${_oplist_yaml}
    --custom-ops-yaml-path=${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml)

set(_gen_command_sources
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterCodegenUnboxedKernelsEverything.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterCPUCustomOps.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterSchema.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/Functions.h
    ${CMAKE_CURRENT_BINARY_DIR}/NativeFunctions.h
    ${CMAKE_CURRENT_BINARY_DIR}/CustomOpsNativeFunctions.h)
message(STATUS "Generating selected operator list ${_gen_oplist_command}")

add_custom_command(
  COMMENT "Generating selected_operators.yaml for custom ops"
  OUTPUT ${_oplist_yaml}
  COMMAND ${_gen_oplist_command}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml ${_codegen_tools_srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT})

add_custom_command(
  COMMENT "Generating code for custom operator registration"
  OUTPUT ${_gen_command_sources}
  COMMAND ${_gen_command}
  DEPENDS ${_oplist_yaml} ${CMAKE_CURRENT_LIST_DIR}/custom_ops.yaml
          ${_codegen_templates} ${_torchgen_srcs}
  WORKING_DIRECTORY ${EXECUTORCH_ROOT})
# Prepare for C++ libraries.

# 1. TODO: C++ library to register custom ops into PyTorch.
# ~~~
# add_library(custom_ops_aot_lib SHARED
#   ${OUTPUT_DIRECTORY}/RegisterCPUCustomOps.cpp
#   ${OUTPUT_DIRECTORY}/RegisterSchema.cpp
#   ${OUTPUT_DIRECTORY}/CustomOpsNativeFunctions.h)
# ~~~

# Find `Torch`.
# ~~~
# find_package(Torch REQUIRED)
# target_link_libraries(custom_ops_aot_lib PUBLIC Torch)
# ~~~

# 1. C++ library to register custom ops into Executorch runtime.

add_library(custom_ops_lib)
target_sources(
  custom_ops_lib
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/RegisterCodegenUnboxedKernelsEverything.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/Functions.h
    ${CMAKE_CURRENT_BINARY_DIR}/NativeFunctions.h
    ${CMAKE_CURRENT_BINARY_DIR}/CustomOpsNativeFunctions.h
    ${CMAKE_CURRENT_LIST_DIR}/custom_ops_1.cpp)

target_link_libraries(custom_ops_lib PRIVATE executorch)

# Ensure that the load-time constructor functions run. By default, the linker
# would remove them since there are no other references to them.
if((CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
   OR (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  target_link_options(custom_ops_lib INTERFACE
                      "-Wl,-force_load,$<TARGET_FILE:custom_ops_lib>")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_link_options(
    custom_ops_lib INTERFACE
    "-Wl,--whole-archive,$<TARGET_FILE:custom_ops_lib>,--no-whole-archive")
endif()

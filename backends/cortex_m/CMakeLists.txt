# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kernel library for Cortex-M operators. Please keep this file formatted by
# running:
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
include(ExternalProject)

# Download and build CMSIS-NN from GitHub
set(CMSIS_NN_VERSION
    "v4.1.0"
    CACHE STRING "CMSIS-NN version to download"
)
set(CMSIS_NN_ROOT ${CMAKE_CURRENT_BINARY_DIR}/cmsis-nn)
set(CMSIS_NN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cmsis-nn-build)
set(CMSIS_NN_LIB_PATH ${CMSIS_NN_BINARY_DIR}/libcmsis-nn.a)

set(TARGET_CPU
    "cortex-m55"
    CACHE STRING "Target CPU for CMSIS-NN build"
)
ExternalProject_Add(
  cmsis_nn_external
  GIT_REPOSITORY https://github.com/ARM-software/CMSIS-NN.git
  GIT_TAG ${CMSIS_NN_VERSION}
  SOURCE_DIR ${CMSIS_NN_ROOT}
  BINARY_DIR ${CMSIS_NN_BINARY_DIR}
  CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${EXECUTORCH_ROOT}/examples/arm/ethos-u-setup/arm-none-eabi-gcc.cmake
    -DTARGET_CPU=${TARGET_CPU}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS ${CMSIS_NN_LIB_PATH}
)

# Cortex-M ops kernel sources
set(_cortex_m_kernels__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_dequantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_add.cpp
)

# Generate C++ bindings to register kernels into Executorch (for runtime). Here
# select all ops in operators.yaml
set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/ops/operators.yaml)
gen_selected_ops(LIB_NAME "cortex_m_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")

# Generate bindings for the kernels
generate_bindings_for_kernels(
  LIB_NAME "cortex_m_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)
message("Generated files ${gen_command_sources}")

# Build a library for _cortex_m_kernels_srcs
add_library(cortex_m_kernels ${_cortex_m_kernels__srcs})
target_compile_options(cortex_m_kernels PUBLIC ${_common_compile_options})

# Add dependency on CMSIS-NN external project
add_dependencies(cortex_m_kernels cmsis_nn_external)

# Set include directories - Include is directly in CMSIS-NN root
target_include_directories(
  cortex_m_kernels
  PRIVATE ${EXECUTORCH_ROOT}/..
          ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
          $<BUILD_INTERFACE:${CMSIS_NN_ROOT}/Include>
          $<BUILD_INTERFACE:${CMSIS_NN_ROOT}>
)

# Link against the CMSIS-NN static library directly
target_link_libraries(cortex_m_kernels PUBLIC ${CMSIS_NN_LIB_PATH} executorch)

# cortex_m_ops_lib: Register Cortex-M ops kernels into Executorch runtime
gen_operators_lib(
  LIB_NAME "cortex_m_ops_lib" KERNEL_LIBS cortex_m_kernels DEPS executorch
)

install(
  TARGETS cortex_m_kernels cortex_m_ops_lib
  EXPORT ExecuTorchTargets
  DESTINATION lib
  PUBLIC_HEADER DESTINATION include/executorch/backends/cortex_m/ops/
)

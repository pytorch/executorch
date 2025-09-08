# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
include(FetchContent)

set(USE_LOCAL_CMSIS_NN OFF)

if(USE_LOCAL_CMSIS_NN)
  if(NOT EXISTS ${CMSIS_NN_LOCAL_PATH})
    message(
      FATAL_ERROR "CMSIS-NN local path does not exist: ${CMSIS_NN_LOCAL_PATH}"
    )
  endif()
  if(NOT EXISTS ${CMSIS_NN_LOCAL_LIB})
    message(
      FATAL_ERROR "CMSIS-NN local lib does not exist: ${CMSIS_NN_LOCAL_LIB}"
    )
  endif()
  message(STATUS "Using CMSIS-NN from: ${CMSIS_NN_LOCAL_PATH}")
  add_subdirectory(${CMSIS_NN_LOCAL_PATH} cmsis_nn_build)
  # Add MVEI define to cmsis-nn target
  target_compile_definitions(cmsis-nn PUBLIC ARM_MATH_MVEI=1)

  set(CMSIS_NN_INCLUDE_DIR "${CMSIS_NN_LOCAL_PATH}/Include")
  set(CMSIS_NN_LIB "${CMSIS_NN_LOCAL_LIB}/libcmsis-nn.a")

else()
  message(STATUS "Using  CMSIS-NN from via : FetchContent")
  set(CMSIS_NN_VERSION
      "v7.0.0"
      CACHE STRING "CMSIS-NN version to download"
  )
  FetchContent_Declare(
    cmsis_nn
    GIT_REPOSITORY https://github.com/ARM-software/CMSIS-NN.git
    GIT_TAG ${CMSIS_NN_VERSION}
  )
  FetchContent_MakeAvailable(cmsis_nn)
  # Add MVEI define to cmsis-nn target
  target_compile_definitions(cmsis-nn PUBLIC ARM_MATH_MVEI=1)
  # Get the correct source and binary dirs
  FetchContent_GetProperties(
    cmsis_nn
    SOURCE_DIR CMSIS_NN_SOURCE_DIR
    BINARY_DIR CMSIS_NN_BINARY_DIR
  )
  set(CMSIS_NN_INCLUDE_DIR "${CMSIS_NN_SOURCE_DIR}/Include")
  set(CMSIS_NN_LIB "${CMSIS_NN_BINARY_DIR}/libcmsis-nn.a")
  message(STATUS " ${CMSIS_NN_SOURCE_DIR}")
  message(STATUS "CMSIS-NN binary dir: ${CMSIS_NN_BINARY_DIR}")
endif()

# Cortex-M ops kernel sources
set(_cortex_m_kernels__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_dequantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_linear.cpp
)

# Generate C++ bindings to register kernels into Executorch (for runtime)
set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/ops/operators.yaml)
gen_selected_ops(LIB_NAME "cortex_m_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")

generate_bindings_for_kernels(
  LIB_NAME "cortex_m_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)
message("Generated files ${gen_command_sources}")

# Build a library for cortex_m_kernels
add_library(cortex_m_kernels ${_cortex_m_kernels__srcs})
target_compile_options(cortex_m_kernels PUBLIC ${_common_compile_options})

# Include directories for cortex_m_kernels
target_include_directories(
  cortex_m_kernels
  PRIVATE ${EXECUTORCH_ROOT}/..
          ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
          ${CMSIS_NN_INCLUDE_DIR}
)

# Link directly to the CMSIS-NN static library file
target_link_libraries(cortex_m_kernels PUBLIC ${CMSIS_NN_LIB} executorch)

# Add dependency to ensure CMSIS-NN builds before we try to link. Use the actual
# CMSIS-NN target name (usually 'cmsis-nn')
add_dependencies(cortex_m_kernels cmsis-nn)

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

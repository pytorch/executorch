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

# Source root directory for executorch
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)
include(FetchContent)

# CMSIS-NN configuration with dynamic path detection
set(CMSIS_NN_VERSION
    "v7.0.0"
    CACHE STRING "CMSIS-NN version to download"
)
set(CMSIS_NN_LOCAL_PATH
    ""
    CACHE PATH "Path to existing local CMSIS-NN installation"
)

# Try to find existing / local CMSIS-NN installation. This is useful for
# debugging and testing with local changes. This is not common, as the CMSIS-NN
# library is downloaded via FetchContent in the default/regular case.
if(CMSIS_NN_LOCAL_PATH AND EXISTS "${CMSIS_NN_LOCAL_PATH}")
  message(STATUS "Using CMSIS-NN from specified path: ${CMSIS_NN_LOCAL_PATH}")
  add_subdirectory(${CMSIS_NN_LOCAL_PATH} _deps/cmsis_nn-build)
else()
  # Use FetchContent with automatic fallback
  message(STATUS "Using CMSIS-NN via FetchContent")

  FetchContent_Declare(
    cmsis_nn
    GIT_REPOSITORY https://github.com/ARM-software/CMSIS-NN.git
    GIT_TAG ${CMSIS_NN_VERSION}
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(cmsis_nn)
endif()

# Cortex-M ops kernel sources
set(_cortex_m_kernels__srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_dequantize_per_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_avg_pool2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_depthwise_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_linear.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_mul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_quantized_transpose_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_minimum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_maximum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/op_transpose.cpp
)

# Generate C++ bindings to register kernels into Executorch
set(_yaml_file ${CMAKE_CURRENT_LIST_DIR}/ops/operators.yaml)
gen_selected_ops(LIB_NAME "cortex_m_ops_lib" OPS_SCHEMA_YAML "${_yaml_file}")
generate_bindings_for_kernels(
  LIB_NAME "cortex_m_ops_lib" CUSTOM_OPS_YAML "${_yaml_file}"
)

# Build library for cortex_m_kernels
add_library(cortex_m_kernels ${_cortex_m_kernels__srcs})

# Use PRIVATE for implementation dependencies to avoid INTERFACE pollution
target_link_libraries(
  cortex_m_kernels
  PRIVATE cmsis-nn
  PRIVATE executorch
  PRIVATE kernels_util_all_deps
)

# Include directories for cortex_m_kernels
target_include_directories(
  cortex_m_kernels PRIVATE ${EXECUTORCH_ROOT}/..
                           ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# cortex_m_ops_lib: Register Cortex-M ops kernels into Executorch runtime
gen_operators_lib(
  LIB_NAME "cortex_m_ops_lib" KERNEL_LIBS cortex_m_kernels DEPS executorch
)

install(
  TARGETS cortex_m_kernels cortex_m_ops_lib cmsis-nn
  EXPORT ExecuTorchTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/executorch/backends/cortex_m/ops/
)

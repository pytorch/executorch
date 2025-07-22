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
set(CMAKE_VERBOSE_MAKEFILE ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable ExecuTorch logging")
set(EXECUTORCH_LOG_LEVEL "DEBUG" CACHE STRING "ExecuTorch log level")

# Path to CMSIS-NN root - adjust as needed
set(CMSIS_NN_ROOT /home/sidart/working/CMSIS-NN)

# Cortex-M CMSIS ops sources
set(_cortex_m_kernels_cmsis__srcs
    "${EXECUTORCH_ROOT}/backends/cortex_m/cmsis-nn/ops/op_aten_add_tensor.cpp"
    "${EXECUTORCH_ROOT}/backends/cortex_m/cmsis-nn/ops/op_aten_softmax.cpp"
)

# Common include directories
set(_common_include_directories
    ${EXECUTORCH_ROOT}/..
    ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
    ${CMSIS_NN_ROOT}/Include
    ${CMSIS_NN_ROOT}  # For any CMake or config includes
)

# Import CMSIS-NN static library as a target
add_library(cmsis_nn STATIC IMPORTED)
set_target_properties(cmsis_nn PROPERTIES
  IMPORTED_LOCATION "${CMSIS_NN_ROOT}/build/libcmsis-nn.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMSIS_NN_ROOT}/Include"
)

# Build cortex_m_cmsis_kernels static library
add_library(cortex_m_cmsis_kernels ${_cortex_m_kernels_cmsis__srcs})

# Include directories for cortex_m_cmsis_kernels
target_include_directories(cortex_m_cmsis_kernels
  PRIVATE
    ${_common_include_directories}
)

# Link libraries: executorch and CMSIS-NN imported target
target_link_libraries(cortex_m_cmsis_kernels
  PRIVATE
    cmsis_nn
    executorch
)

# Generate C++ bindings for kernels and operators
gen_selected_ops(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" OPS_SCHEMA_YAML
  "${CMAKE_CURRENT_LIST_DIR}/../cmsis.yaml" "" ""
)
generate_bindings_for_kernels(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" FUNCTIONS_YAML
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmsis.yaml
)

gen_operators_lib(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" KERNEL_LIBS cortex_m_cmsis_kernels DEPS executorch
)
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--gc-sections")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")

# Install targets and headers
install(
  TARGETS cortex_m_cmsis_kernels cortex_m_cmsis_nn_ops_lib
  DESTINATION lib
  PUBLIC_HEADER DESTINATION include/executorch/backends/cortex_m/cmsis-nn/ops/
)

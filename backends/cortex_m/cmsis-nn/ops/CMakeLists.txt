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

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable ExecuTorch logging")
set(EXECUTORCH_LOG_LEVEL "DEBUG" CACHE STRING "ExecuTorch log level")

# Cortex-M CMSIS ops that are needed to run this model.
set(_cortex_m_kernels_cmsis__srcs
    "${EXECUTORCH_ROOT}/backends/cortex_m/cmsis-nn/ops/op_aten_add_tensor.cpp"
    "${EXECUTORCH_ROOT}/backends/cortex_m/cmsis-nn/ops/op_aten_softmax.cpp"
    )

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories ${EXECUTORCH_ROOT}/..
${EXECUTORCH_ROOT}/runtime/core/portable_type/c10)

add_library(cortex_m_cmsis_kernels ${_cortex_m_kernels_cmsis__srcs})
target_link_libraries(cortex_m_cmsis_kernels PRIVATE executorch)
target_compile_options(cortex_m_cmsis_kernels PUBLIC ${_common_compile_options})

# Generate C++ bindings to register kernels into both PyTorch (for AOT) and
# Executorch (for runtime). Here select all ops in functions.yaml
gen_selected_ops(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" OPS_SCHEMA_YAML
  "${CMAKE_CURRENT_LIST_DIR}/../cmsis.yaml" "" ""
)
generate_bindings_for_kernels(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" FUNCTIONS_YAML
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmsis.yaml
)
message("Generated files ${gen_command_sources}")

gen_operators_lib(
  LIB_NAME "cortex_m_cmsis_nn_ops_lib" KERNEL_LIBS cortex_m_cmsis_kernels DEPS executorch
)

install(
  TARGETS cortex_m_cmsis_kernels cortex_m_cmsis_nn_ops_lib
  DESTINATION lib
  PUBLIC_HEADER DESTINATION include/executorch/backends/cortex_m/cmsis-nn/ops/
)

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
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

if(NOT PYTHON_EXECUTABLE)
  resolve_python_executable()
endif()

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  # Merge optimized and portable definitions, taking optimized where available.
  merge_yaml(
    FUNCTIONS_YAML ${EXECUTORCH_ROOT}/kernels/optimized/optimized-oss.yaml
    FALLBACK_YAML ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml OUTPUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}
  )

  gen_selected_ops(
    LIB_NAME "optimized_native_cpu_ops_lib" OPS_SCHEMA_YAML
    "${CMAKE_CURRENT_BINARY_DIR}/merged.yaml"
  )

  generate_bindings_for_kernels(
    LIB_NAME "optimized_native_cpu_ops_lib" FUNCTIONS_YAML
    ${CMAKE_CURRENT_BINARY_DIR}/merged.yaml
  )
  message("Generated files ${gen_command_sources}")

  # optimized_native_cpu_ops_lib: Register optimized op kernels into the runtime
  gen_operators_lib(
    LIB_NAME
    "optimized_native_cpu_ops_lib"
    KERNEL_LIBS
    portable_kernels
    optimized_kernels
    DEPS
    executorch
  )

  install(TARGETS optimized_native_cpu_ops_lib DESTINATION lib)
endif()

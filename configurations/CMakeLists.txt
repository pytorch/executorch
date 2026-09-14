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

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_common_compile_options -Wno-deprecated-declarations)

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Codegen.cmake)

if(EXECUTORCH_BUILD_KERNELS_OPTIMIZED)
  # Merge optimized and portable definitions, taking optimized where available.
  merge_yaml(
    FUNCTIONS_YAML ${EXECUTORCH_ROOT}/kernels/optimized/optimized.yaml
    FALLBACK_YAML ${EXECUTORCH_ROOT}/kernels/portable/functions.yaml OUTPUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}
  )

  gen_selected_ops(
    LIB_NAME "optimized_native_cpu_ops_lib" OPS_SCHEMA_YAML
    "${CMAKE_CURRENT_BINARY_DIR}/merged.yaml"
  )

  generate_bindings_for_kernels(
    LIB_NAME "optimized_native_cpu_ops_lib" FUNCTIONS_YAML
    ${CMAKE_CURRENT_BINARY_DIR}/merged.yaml ADD_EXCEPTION_BOUNDARY
  )
  message("Generated files ${gen_command_sources}")

  # optimized_native_cpu_ops_lib: Register optimized op kernels into the runtime
  if(TARGET optimized_portable_kernels)
    set(_optimized_native_cpu_ops_lib_portable_kernels_lib
        optimized_portable_kernels
    )
  else()
    set(_optimized_native_cpu_ops_lib_portable_kernels_lib portable_kernels)
  endif()
  # Ship this as a shared library in the wheel so the kernels are registered
  # once per process instead of once per component that links them.
  if(EXECUTORCH_BUILD_SHARED)
    set(_merged_cpu_ops_library_type SHARED)
  else()
    set(_merged_cpu_ops_library_type "")
  endif()
  gen_operators_lib(
    ${_merged_cpu_ops_library_type}
    LIB_NAME
    "optimized_native_cpu_ops_lib"
    KERNEL_LIBS
    optimized_kernels
    ${_optimized_native_cpu_ops_lib_portable_kernels_lib}
    DEPS
    executorch_core
  )

  install(
    TARGETS optimized_native_cpu_ops_lib
    EXPORT ExecuTorchTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  if(EXECUTORCH_BUILD_SHARED)
    # Named after what the library provides rather than after the code
    # generation target that produces it, so the shipped file reads as
    # libexecutorch_kernels_optimized.so. Only for the wheel: a source install
    # keeps the existing file name, which projects already link against.
    if(EXECUTORCH_BUILD_WHEEL_DO_NOT_USE)
      set_target_properties(
        optimized_native_cpu_ops_lib PROPERTIES OUTPUT_NAME
                                                executorch_kernels_optimized
      )
    endif()
    executorch_target_soname_policy(optimized_native_cpu_ops_lib)
  endif()
endif()

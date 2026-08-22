# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

# Disable install(rule) generation while building under Zephyr. The sample
# application never runs `cmake --install`, and skipping the rules avoids export
# dependency checks when Zephyr provides targets such as ethosu_core_driver.
set(CMAKE_SKIP_INSTALL_RULES
    ON
    CACHE BOOL "" FORCE
)

# Use configuration overlay to disable default portable ops to enable selective
# operator build
set(ET_PTE_FILE_PATH
    ""
    CACHE FILEPATH "Path to the ExecuTorch .pte (or .bpte) model to embed"
)
set(ET_PTE_SECTION
    "network_model_sec"
    CACHE STRING "Section attribute used for the generated model data"
)

if(NOT ET_PTE_FILE_PATH)
  message(
    FATAL_ERROR
      "ET_PTE_FILE_PATH must point to the ExecuTorch .pte (or .bpte) model to embed."
  )
endif()

if(NOT IS_ABSOLUTE "${ET_PTE_FILE_PATH}")
  get_filename_component(
    ET_PTE_FILE_PATH "${ET_PTE_FILE_PATH}" ABSOLUTE BASE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()

if(NOT EXISTS "${ET_PTE_FILE_PATH}")
  message(
    FATAL_ERROR
      "Could not find ExecuTorch model at ET_PTE_FILE_PATH: ${ET_PTE_FILE_PATH}"
  )
endif()

set(ET_PTE_FILE_PATH
    "${ET_PTE_FILE_PATH}"
    CACHE FILEPATH "Path to the ExecuTorch .pte (or .bpte) model to embed"
          FORCE
)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(executorch_executor_runner)

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED
)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wno-switch -Wno-float-conversion -Wno-double-promotion -ffunction-sections -fdata-sections"
)

# Always use selective build since embedded devices are resource constrained

# Include ExecuTorch selective build utilities Use EXECUTORCH_DIR if available,
# otherwise use Zephyr's module discovery
if(NOT DEFINED EXECUTORCH_DIR)
  message(
    STATUS
      "ZEPHYR_EXECUTORCH_MODULE_DIR set to : ${ZEPHYR_EXECUTORCH_MODULE_DIR}"
  )
  if(DEFINED ZEPHYR_EXECUTORCH_MODULE_DIR)
    set(EXECUTORCH_DIR ${ZEPHYR_EXECUTORCH_MODULE_DIR})
    message(
      STATUS "Using Zephyr module discovery: EXECUTORCH_DIR=${EXECUTORCH_DIR}"
    )
  else()
    message(
      FATAL_ERROR
        "ExecuTorch module not found. Ensure it's properly configured in your Zephyr workspace."
    )
  endif()
else()
  message(STATUS "Using predefined EXECUTORCH_DIR=${EXECUTORCH_DIR}")
endif()

include(${EXECUTORCH_DIR}/tools/cmake/Utils.cmake)
include(${EXECUTORCH_DIR}/backends/arm/cmake/ArmRunnerUtils.cmake)

# Ensure the portable kernels target exists even when portable ops are disabled
# in the global build.
if(NOT TARGET portable_kernels)
  set(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY ON)
  add_subdirectory(
    ${EXECUTORCH_DIR}/kernels/portable
    ${CMAKE_CURRENT_BINARY_DIR}/executorch/kernels/portable
  )
  unset(EXECUTORCH_PORTABLE_BUILD_KERNELS_ONLY)
endif()
set(_executorch_zephyr_selected_ops_libs)
if(NOT DEFINED EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD)
  set(EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD "")
endif()

arm_runner_create_selected_ops_lib(
  LIB_NAME
  "cpu_portable_ops_lib"
  FUNCTIONS_YAML
  "${EXECUTORCH_DIR}/kernels/portable/functions.yaml"
  OPS_FROM_MODEL
  "${ET_PTE_FILE_PATH}"
  KERNEL_LIBS
  portable_kernels
  DEPS
  executorch
  DTYPE_SELECTIVE_BUILD
  "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
)
list(APPEND _executorch_zephyr_selected_ops_libs cpu_portable_ops_lib)

if(TARGET cortex_m_kernels)
  arm_runner_create_selected_ops_lib(
    LIB_NAME
    "zephyr_cortex_m_ops_lib"
    CUSTOM_OPS_YAML
    "${EXECUTORCH_DIR}/backends/cortex_m/ops/operators.yaml"
    OPS_FROM_MODEL
    "${ET_PTE_FILE_PATH}"
    KERNEL_LIBS
    cortex_m_kernels
    DEPS
    executorch
  )
  list(APPEND _executorch_zephyr_selected_ops_libs zephyr_cortex_m_ops_lib)
endif()

set(_local_flatcc_root ${CMAKE_BINARY_DIR}/flatcc_src)
if(NOT EXISTS ${_local_flatcc_root}/CMakeLists.txt)
  file(MAKE_DIRECTORY ${_local_flatcc_root})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${EXECUTORCH_DIR}/third-party/flatcc ${_local_flatcc_root}
  )
endif()
set(EXECUTORCH_FLATCC_SOURCE_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)
set(EXECUTORCH_FLATCC_INSTALL_ROOT
    ${_local_flatcc_root}
    CACHE PATH "" FORCE
)

set(app_sources
    src/arm_executor_runner.cpp
    ${EXECUTORCH_DIR}/examples/arm/executor_runner/arm_memory_allocator.cpp
)
target_sources(app PRIVATE ${app_sources})

set(_model_pte_header ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h)
add_custom_command(
  OUTPUT ${_model_pte_header}
  COMMAND
    ${Python3_EXECUTABLE}
    ${EXECUTORCH_DIR}/examples/arm/executor_runner/pte_to_header.py --pte
    ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR} --section
    ${ET_PTE_SECTION}
  DEPENDS ${ET_PTE_FILE_PATH}
          ${EXECUTORCH_DIR}/examples/arm/executor_runner/pte_to_header.py
  COMMENT "Converting ${ET_PTE_FILE_PATH} to model_pte.h"
)
add_custom_target(gen_model_header DEPENDS ${_model_pte_header})
add_dependencies(app gen_model_header)

if(DEFINED CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    app
    PRIVATE
      ET_ARM_METHOD_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_METHOD_ALLOCATOR_POOL_SIZE}
  )
endif()
if(DEFINED CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    app
    PRIVATE
      ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${CONFIG_EXECUTORCH_TEMP_ALLOCATOR_POOL_SIZE}
  )
endif()

# Link with ExecuTorch and our operators library
target_link_libraries(app PRIVATE libexecutorch)
target_link_libraries(app PRIVATE ${_executorch_zephyr_selected_ops_libs})
if(TARGET executorch_delegate_ethos_u)
  executorch_target_link_options_shared_lib(executorch_delegate_ethos_u)
  target_link_libraries(app PRIVATE executorch_delegate_ethos_u)
endif()
if(TARGET ethosu_core_driver)
  target_link_libraries(app PRIVATE ethosu_core_driver)
endif()

# Add include directories for sources and generated headers
target_include_directories(app PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
get_target_property(OUT app LINK_LIBRARIES)
message(STATUS ${OUT})

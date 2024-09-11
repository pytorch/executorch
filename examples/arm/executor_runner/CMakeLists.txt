# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_executor_runner)

option(SEMIHOSTING "Enable semihosting" OFF)

if(NOT DEFINED ET_PTE_FILE_PATH AND NOT ${SEMIHOSTING})
  message(
    FATAL_ERROR
      "ET_PTE_FILE_PATH must specify a model .pte, for bare metal systems the "
      "model is built into the binary."
  )
endif()

# Example ExecuTorch demo for bare metal Cortex-M based systems
set(ET_DIR_PATH
    "../../.."
    CACHE PATH "Path to ExecuTorch dir"
)
set(ET_BUILD_DIR_PATH
    "${ET_DIR_PATH}/cmake-out"
    CACHE PATH "Path to ExecuTorch build dir"
)
set(ET_INCLUDE_PATH
    "${ET_DIR_PATH}/.."
    CACHE PATH "Path to ExecuTorch headers"
)
set(ET_PTE_FILE_PATH
    ""
    CACHE PATH "Path to ExecuTorch model pte"
)
set(ETHOS_SDK_PATH
    "${ET_DIR_PATH}/examples/arm/ethos-u-scratch/ethos-u"
    CACHE PATH "Path to Ethos-U bare metal driver/env"
)
set(PYTHON_EXECUTABLE
    "python"
    CACHE PATH "Define to override python executable used"
)

get_filename_component(ET_BUILD_DIR_PATH ${ET_BUILD_DIR_PATH} REALPATH)
get_filename_component(ET_DIR_PATH ${ET_DIR_PATH} REALPATH)
get_filename_component(ET_INCLUDE_PATH ${ET_INCLUDE_PATH} REALPATH)
get_filename_component(ETHOS_SDK_PATH ${ETHOS_SDK_PATH} REALPATH)
if(NOT ${SEMIHOSTING})
  get_filename_component(ET_PTE_FILE_PATH ${ET_PTE_FILE_PATH} REALPATH)
endif()

# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.

# For Corstone-300 FVP builds we put models into the larger DRAM area
set(MEMORY_MODEL "dram")
set(MEMORY_ARENA "dram")
add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target)

# Dependencies from the ExecuTorch build
add_library(executorch STATIC IMPORTED)
set_property(
  TARGET executorch PROPERTY IMPORTED_LOCATION
                             "${ET_BUILD_DIR_PATH}/libexecutorch.a"
)

add_library(executorch_no_prim_ops STATIC IMPORTED)
set_property(
  TARGET executorch_no_prim_ops
  PROPERTY IMPORTED_LOCATION "${ET_BUILD_DIR_PATH}/libexecutorch_no_prim_ops.a"
)
target_link_libraries(executorch INTERFACE executorch_no_prim_ops)

add_library(executorch_delegate_ethos_u STATIC IMPORTED)
set_property(
  TARGET executorch_delegate_ethos_u
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/backends/arm/libexecutorch_delegate_ethos_u.a"
)

add_library(portable_ops_lib STATIC IMPORTED)
set_property(
  TARGET portable_ops_lib
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/examples/arm/libarm_portable_ops_lib.a"
)
add_library(portable_kernels STATIC IMPORTED)
set_property(
  TARGET portable_kernels
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/kernels/portable/libportable_kernels.a"
)

add_library(quantized_ops_lib STATIC IMPORTED)
set_property(
  TARGET quantized_ops_lib
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/kernels/quantized/libquantized_ops_lib.a"
)
add_library(quantized_kernels STATIC IMPORTED)
set_property(
  TARGET quantized_kernels
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/kernels/quantized/libquantized_kernels.a"
)

add_library(extension_runner_util STATIC IMPORTED)
set_property(
  TARGET extension_runner_util
  PROPERTY
    IMPORTED_LOCATION
    "${ET_BUILD_DIR_PATH}/extension/runner_util/libextension_runner_util.a"
)

# Convert pte to header
if(NOT ${SEMIHOSTING})
  add_custom_target(
    gen_model_header DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/model_pte.h
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/pte_to_header.py --pte
            ${ET_PTE_FILE_PATH} --outdir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ET_PTE_FILE_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

# The arm_executor_runner executable
add_executable(arm_executor_runner)

target_sources(
  arm_executor_runner PRIVATE arm_executor_runner.cpp arm_perf_monitor.cpp
)

# Include the target's bare-metal linker script
ethosu_eval_link_options(arm_executor_runner)

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(
  arm_executor_runner
  extension_runner_util
  ethosu_target_init
  executorch
  "-Wl,--whole-archive"
  executorch_delegate_ethos_u
  quantized_ops_lib
  portable_ops_lib
  quantized_kernels
  portable_kernels
  "-Wl,--no-whole-archive"
)

# ET headers and generated headers includes
target_include_directories(
  arm_executor_runner PRIVATE ${ET_INCLUDE_PATH} ${CMAKE_CURRENT_BINARY_DIR}
)

if(SEMIHOSTING)
  target_compile_definitions(arm_executor_runner PUBLIC SEMIHOSTING)
else()
  add_dependencies(arm_executor_runner gen_model_header)
endif()

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/corstone-300/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()

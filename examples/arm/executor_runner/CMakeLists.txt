# Copyright 2023-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_executor_runner)

option(SEMIHOSTING "Enable semihosting" OFF)
option(
  ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE
  "Set ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE to specify memory alloction pool size"
  OFF
)
option(ET_BUNDLE_IO "Set to compile in BundleIO support" OFF)
option(ET_ATOL "Set atol to use for BundleIO testing" OFF)
option(ET_RTOL "Set rtol to use for BundleIO testing" OFF)
option(ET_DUMP_INPUT "Dump input in log" OFF)
option(ET_DUMP_OUTPUT "Dump output in log" ON)
option(FETCH_ETHOS_U_CONTENT
       "Fetch ethos_u dependencies instead of relying on pre-downloads" ON
)
set(ET_NUM_INFERENCES
    "1"
    CACHE STRING "Number of inferences to run"
)

if(NOT DEFINED ET_PTE_FILE_PATH AND NOT ${SEMIHOSTING})
  message(
    FATAL_ERROR
      "ET_PTE_FILE_PATH must specify a model .pte or .bpte, for bare metal systems the "
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

# Include corstone help functions
include(${ET_DIR_PATH}/backends/arm/scripts/corstone_utils.cmake)

if(FETCH_ETHOS_U_CONTENT)
  # Download ethos_u dependency if needed.
  fetch_ethos_u_content(${ETHOS_SDK_PATH} ${ET_DIR_PATH})
endif()

# Selects timing adapter values matching system_config. Default is
# Ethos_U55_High_End_Embedded, simulating optimal hardware for the
# Corestone-300.
set(SYSTEM_CONFIG
    "Ethos_U55_High_End_Embedded"
    CACHE STRING "System config"
)
set(MEMORY_MODE
    "Shared_Sram"
    CACHE STRING "Vela memory mode"
)

message(STATUS "SYSTEM_CONFIG is ${SYSTEM_CONFIG}")
message(STATUS "MEMORY_MODE is ${MEMORY_MODE}")
message(STATUS "ET_NUM_INFERENCES is ${ET_NUM_INFERENCES}")

get_filename_component(ET_BUILD_DIR_PATH ${ET_BUILD_DIR_PATH} REALPATH)
get_filename_component(ET_DIR_PATH ${ET_DIR_PATH} REALPATH)
get_filename_component(ET_INCLUDE_PATH ${ET_INCLUDE_PATH} REALPATH)
get_filename_component(ETHOS_SDK_PATH ${ETHOS_SDK_PATH} REALPATH)
if(NOT ${SEMIHOSTING})
  get_filename_component(ET_PTE_FILE_PATH ${ET_PTE_FILE_PATH} REALPATH)
endif()

# By default, use 2MB of temporary scratch buffer For Dedicated_Sram, use 64MB
# for the temporary scratch buffer and 384KB for the fast scratch buffer(the
# cache, applicable only for Ethos-U65 and Ethos-U85)
set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x200000)
if(MEMORY_MODE MATCHES "Dedicated_Sram")
  set(ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x4000000)
  set(ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE 0x60000)
endif()
message(
  STATUS
    "ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE = ${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}"
)
message(
  STATUS
    "ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE = ${ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}"
)

# Dependencies from the Ethos-U Core This is the platform target of
# Corstone-300, that includes ethosu_core_driver and bare-metal bringup
# libraries. We link against ethosu_target_init which includes all of these
# dependencies.
add_corstone_subdirectory(${SYSTEM_CONFIG} ${ETHOS_SDK_PATH})
configure_timing_adapters(${SYSTEM_CONFIG} ${MEMORY_MODE})

# Dependencies from the ExecuTorch build
add_library(executorch STATIC IMPORTED)
set_property(
  TARGET executorch PROPERTY IMPORTED_LOCATION
                           "${ET_BUILD_DIR_PATH}/libprim_ops_lib.a"
)

add_library(executorch_core STATIC IMPORTED)
set_property(
  TARGET executorch_core
  PROPERTY IMPORTED_LOCATION "${ET_BUILD_DIR_PATH}/libexecutorch_core.a"
)
target_link_libraries(executorch INTERFACE executorch_core)

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
add_library(cortex_m_ops_lib STATIC IMPORTED)
set_property(
  TARGET cortex_m_ops_lib
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/backends/cortex_m/libcortex_m_ops_lib.a"
)
add_library(cortex_m_kernels STATIC IMPORTED)
set_property(
  TARGET cortex_m_kernels
  PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/backends/cortex_m/libcortex_m_kernels.a"
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
                              arm_memory_allocator.cpp
)

# Include the target's bare-metal linker script
ethosu_eval_link_options(arm_executor_runner)

set(arm_executor_runner_link)
list(
  APPEND
  arm_executor_runner_link
  extension_runner_util
  ethosu_target_init
  executorch
  quantized_ops_lib
  cortex_m_ops_lib
  "-Wl,--whole-archive"
  executorch_delegate_ethos_u
  quantized_kernels
  cortex_m_kernels
  portable_kernels
  "-Wl,--no-whole-archive"
  -Xlinker
  -Map=arm_executor_runner.map
)

# Prefer to generate kernel bindings from model file if possible, which is when
# 1. Not building for semihosting 2. Not building with bundleio If that is not
# the case, fallback to select_ops_list If the model file does not contain any
# aten ops, a workaround is currently needed to avoid crashing.
execute_process(
  COMMAND
    python "${ET_DIR_PATH}/codegen/tools/gen_oplist.py"
    --model_file_path=${ET_PTE_FILE_PATH}
    --output_path=${CURRENT_BINARY_DIR}/temp.yaml
  OUTPUT_VARIABLE CMD_RESULT
)
if(NOT CMD_RESULT MATCHES "aten::")
  set(NO_OPS_IN_FILE "true")
else()
  set(NO_OPS_IN_FILE "false")
endif()

if(${SEMIHOSTING} OR ${ET_BUNDLE_IO})
  set(EXECUTORCH_SELECT_OPS_MODEL "")
elseif(${NO_OPS_IN_FILE})
  set(EXECUTORCH_SELECT_OPS_LIST "")
  set(EXECUTORCH_SELECT_OPS_MODEL "")
else()
  set(EXECUTORCH_SELECT_OPS_LIST "")
  set(EXECUTORCH_SELECT_OPS_MODEL "${ET_PTE_FILE_PATH}")
endif()

# Ensure that either executorch_select_ops_list or executorch_select_ops_model
# is set - otherwise assume no kernels needs to be registered
if(NOT ("${EXECUTORCH_SELECT_OPS_LIST}" STREQUAL ""
        AND "${EXECUTORCH_SELECT_OPS_MODEL}" STREQUAL "")
)
  set(EXECUTORCH_ROOT ${ET_DIR_PATH})
  include(${ET_DIR_PATH}/tools/cmake/Utils.cmake)
  include(${ET_DIR_PATH}/tools/cmake/Codegen.cmake)

  gen_selected_ops(
    LIB_NAME
    "arm_portable_ops_lib"
    OPS_SCHEMA_YAML
    ""
    ROOT_OPS
    "${EXECUTORCH_SELECT_OPS_LIST}"
    INCLUDE_ALL_OPS
    ""
    OPS_FROM_MODEL
    "${EXECUTORCH_SELECT_OPS_MODEL}"
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )

  generate_bindings_for_kernels(
    LIB_NAME "arm_portable_ops_lib" FUNCTIONS_YAML
    ${ET_DIR_PATH}/kernels/portable/functions.yaml DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )
  gen_operators_lib(
    LIB_NAME
    "arm_portable_ops_lib"
    KERNEL_LIBS
    portable_kernels
    DEPS
    executorch
    DTYPE_SELECTIVE_BUILD
    "${EXECUTORCH_ENABLE_DTYPE_SELECTIVE_BUILD}"
  )
  list(APPEND arm_executor_runner_link arm_portable_ops_lib)
endif()

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  target_compile_options(arm_executor_runner PUBLIC -DET_EVENT_TRACER_ENABLED)

  list(APPEND arm_executor_runner_link etdump flatccrt)
endif()

if(ET_BUNDLE_IO)
  list(APPEND arm_executor_runner_link bundled_program)
endif()

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(arm_executor_runner ${arm_executor_runner_link})

target_link_options(
  arm_executor_runner PUBLIC LINKER:-Map=arm_executor_runner.map
)

# ET headers and generated headers includes
target_include_directories(
  arm_executor_runner
  PRIVATE ${ET_INCLUDE_PATH} ${ET_DIR_PATH}/runtime/core/portable_type/c10
          ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_definitions(
  arm_executor_runner PRIVATE C10_USING_CUSTOM_GENERATED_MACROS
)

if(SEMIHOSTING)
  target_compile_definitions(arm_executor_runner PUBLIC SEMIHOSTING)
else()
  add_dependencies(arm_executor_runner gen_model_header)
endif()

if(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE}
  )
endif()

target_compile_definitions(
  arm_executor_runner
  PUBLIC
    ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
)
if(DEFINED ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(
    arm_executor_runner
    PUBLIC
      ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_FAST_SCRATCH_TEMP_ALLOCATOR_POOL_SIZE}
  )
endif()

if(ET_BUNDLE_IO)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_BUNDLE_IO)
endif()

if(ET_ATOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_ATOL=${ET_ATOL})
endif()

if(ET_RTOL)
  target_compile_definitions(arm_executor_runner PUBLIC ET_RTOL=${ET_RTOL})
endif()

if(ET_DUMP_INPUT)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_DUMP_INPUT)
endif()

if(ET_DUMP_OUTPUT)
  target_compile_definitions(arm_executor_runner PUBLIC -DET_DUMP_OUTPUT)
endif()

if(ET_NUM_INFERENCES)
  target_compile_definitions(
    arm_executor_runner PUBLIC ET_NUM_INFERENCES=${ET_NUM_INFERENCES}
  )
endif()

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/${TARGET_BOARD}/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()

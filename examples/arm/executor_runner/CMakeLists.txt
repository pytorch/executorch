# Copyright 2023-2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.20)
project(arm_executor_runner)

option(SEMIHOSTING "Enable semihosting" OFF)
option(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE "Set ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE to specify memory alloction pool size" OFF)
option(ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE "Set ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE to specify temp alloction pool size" OFF)

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

# Selects timing adapter values matching system_config.
# Default is Ethos_U55_High_End_Embedded, simulating optimal hardware for the Corestone-300.
set(SYSTEM_CONFIG "Ethos_U55_High_End_Embedded" CACHE STRING "System config")

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
if(SYSTEM_CONFIG STREQUAL "Ethos_U55_High_End_Embedded")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target)
  set(TARGET_BOARD "corstone-300")
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=32
      ETHOSU_TA_WLATENCY_0=32
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # Flash
      ETHOSU_TA_MAXR_1=2
      ETHOSU_TA_MAXW_1=0
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=64
      ETHOSU_TA_WLATENCY_1=0
      ETHOSU_TA_PULSE_ON_1=320
      ETHOSU_TA_PULSE_OFF_1=80
      ETHOSU_TA_BWCAP_1=50
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      )
elseif(SYSTEM_CONFIG STREQUAL "Ethos_U55_Deep_Embedded")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target)
  set(TARGET_BOARD "corstone-300")
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=4
      ETHOSU_TA_MAXW_0=4
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=8
      ETHOSU_TA_WLATENCY_0=8
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # Flash
      ETHOSU_TA_MAXR_1=2
      ETHOSU_TA_MAXW_1=0
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=32
      ETHOSU_TA_WLATENCY_1=0
      ETHOSU_TA_PULSE_ON_1=360
      ETHOSU_TA_PULSE_OFF_1=40
      ETHOSU_TA_BWCAP_1=25
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      )
elseif(SYSTEM_CONFIG STREQUAL "Ethos_U85_SYS_DRAM_Low")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)
  set(TARGET_BOARD "corstone-320")
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=16
      ETHOSU_TA_WLATENCY_0=16
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # DRAM
      ETHOSU_TA_MAXR_1=24
      ETHOSU_TA_MAXW_1=12
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=250
      ETHOSU_TA_WLATENCY_1=125
      ETHOSU_TA_PULSE_ON_1=4000
      ETHOSU_TA_PULSE_OFF_1=1000
      ETHOSU_TA_BWCAP_1=2344
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      )
elseif(SYSTEM_CONFIG STREQUAL "Ethos_U85_SYS_DRAM_Mid" OR SYSTEM_CONFIG STREQUAL "Ethos_U85_SYS_DRAM_High")
  add_subdirectory(${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target)
  set(TARGET_BOARD "corstone-320")
  target_compile_definitions(ethosu_target_common INTERFACE
      # ETHOSU_MODEL=0 place pte file/data in SRAM area
      # ETHOSU_MODEL=1 place pte file/data in DDR area
      ETHOSU_MODEL=1
      # Configure NPU architecture timing adapters
      # This is just example numbers and you should make this match your hardware
      # SRAM
      ETHOSU_TA_MAXR_0=8
      ETHOSU_TA_MAXW_0=8
      ETHOSU_TA_MAXRW_0=0
      ETHOSU_TA_RLATENCY_0=32
      ETHOSU_TA_WLATENCY_0=32
      ETHOSU_TA_PULSE_ON_0=3999
      ETHOSU_TA_PULSE_OFF_0=1
      ETHOSU_TA_BWCAP_0=4000
      ETHOSU_TA_PERFCTRL_0=0
      ETHOSU_TA_PERFCNT_0=0
      ETHOSU_TA_MODE_0=1
      ETHOSU_TA_HISTBIN_0=0
      ETHOSU_TA_HISTCNT_0=0
      # DRAM
      ETHOSU_TA_MAXR_1=64
      ETHOSU_TA_MAXW_1=32
      ETHOSU_TA_MAXRW_1=0
      ETHOSU_TA_RLATENCY_1=500
      ETHOSU_TA_WLATENCY_1=250
      ETHOSU_TA_PULSE_ON_1=4000
      ETHOSU_TA_PULSE_OFF_1=1000
      ETHOSU_TA_BWCAP_1=3750
      ETHOSU_TA_PERFCTRL_1=0
      ETHOSU_TA_PERFCNT_1=0
      ETHOSU_TA_MODE_1=1
      ETHOSU_TA_HISTBIN_1=0
      ETHOSU_TA_HISTCNT_1=0
      )
else()
  message(FATAL_ERROR "Unsupported SYSTEM_CONFIG: ${SYSTEM_CONFIG}")
endif()

# Dependencies from the ExecuTorch build
add_library(executorch STATIC IMPORTED)
set_property(
  TARGET executorch PROPERTY IMPORTED_LOCATION
                             "${ET_BUILD_DIR_PATH}/libexecutorch.a"
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

set(arm_executor_runner_link)
list(APPEND arm_executor_runner_link
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
  -Xlinker -Map=arm_executor_runner.map
)

if(EXECUTORCH_ENABLE_EVENT_TRACER)
  target_compile_options(arm_executor_runner PUBLIC -DET_EVENT_TRACER_ENABLED)

  add_library(etdump STATIC IMPORTED)
  set_property(
      TARGET etdump
      PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/lib/libetdump.a"
  )

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(FLATCCRT_LIB flatccrt_d)
  else()
    set(FLATCCRT_LIB flatccrt)
  endif()

  add_library(${FLATCCRT_LIB} STATIC IMPORTED)
  set_property(
      TARGET ${FLATCCRT_LIB}
      PROPERTY IMPORTED_LOCATION
           "${ET_BUILD_DIR_PATH}/lib/lib${FLATCCRT_LIB}.a"
  )

  list(APPEND arm_executor_runner_link
    etdump
    ${FLATCCRT_LIB}
  )
endif()

# Need whole-archive to ensure C++ ctor's are called - this may be wasteful for
# bin size as we link in a number of other symbols
target_link_libraries(
  arm_executor_runner
  ${arm_executor_runner_link}
)

target_link_options( arm_executor_runner PUBLIC LINKER:-Map=arm_executor_runner.map )

# ET headers and generated headers includes
target_include_directories(
  arm_executor_runner PRIVATE ${ET_INCLUDE_PATH} ${CMAKE_CURRENT_BINARY_DIR}
)

if(SEMIHOSTING)
  target_compile_definitions(arm_executor_runner PUBLIC SEMIHOSTING)
else()
  add_dependencies(arm_executor_runner gen_model_header)
endif()

if(ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(arm_executor_runner PUBLIC ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_METHOD_ALLOCATOR_POOL_SIZE})
endif()

if(ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE)
  target_compile_definitions(arm_executor_runner PUBLIC ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE=${ET_ARM_BAREMETAL_TEMP_ALLOCATOR_POOL_SIZE})
endif()

# Fixup compilation of retarget.c
if(SEMIHOSTING)
  # Remove this when MLBEDSW-8910 is closed.
  set_source_files_properties(
    ${ETHOS_SDK_PATH}/core_platform/targets/${TARGET_BOARD}/retarget.c
    PROPERTIES HEADER_FILE_ONLY TRUE
  )
endif()

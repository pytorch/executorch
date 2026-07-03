# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(patch_ethos_u_repo REPO_PATH BASE_REV PATCH_DIR ET_DIR_PATH)
  execute_process(
    COMMAND
      bash -c
      "source backends/arm/scripts/utils.sh && patch_repo \"$1\" \"$2\" \"$3\""
      patch_ethos_u_repo "${REPO_PATH}" "${BASE_REV}" "${PATCH_DIR}"
    WORKING_DIRECTORY "${ET_DIR_PATH}"
    RESULT_VARIABLE patch_result
  )
  if(patch_result)
    message(
      FATAL_ERROR "Failed to apply Ethos-U setup patches to ${REPO_PATH}."
    )
  endif()
endfunction()

function(fetch_ethos_u_content ETHOS_SDK_PATH ET_DIR_PATH)
  message(STATUS "Fetching Ethos-U content into ${ETHOS_SDK_PATH}")

  file(MAKE_DIRECTORY ${ETHOS_SDK_PATH}/../ethos_u)
  include(FetchContent)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(ethos_u_base_tag "26.02")
  FetchContent_Declare(
    ethos_u
    GIT_REPOSITORY
      https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u.git
    GIT_TAG ${ethos_u_base_tag}
    SOURCE_DIR
    ${ETHOS_SDK_PATH}
    BINARY_DIR
    ${ETHOS_SDK_PATH}
    SUBBUILD_DIR
    ${ETHOS_SDK_PATH}/../ethos_u-subbuild
    SOURCE_SUBDIR
    none
  )
  FetchContent_MakeAvailable(ethos_u)
  # Patch manifest to remove unused projects.
  set(patch_dir "${ET_DIR_PATH}/examples/arm/ethos-u-setup")
  set(ethos_u_base_rev "26.02")
  patch_ethos_u_repo(
    "${ETHOS_SDK_PATH}" "${ethos_u_base_rev}" "${patch_dir}" "${ET_DIR_PATH}"
  )

  # Get ethos_u externals only if core driver headers do not already exist.
  if(NOT EXISTS
     "${ETHOS_SDK_PATH}/core_software/core_driver/include/ethosu_driver.h"
  )
    execute_process(
      COMMAND ${Python3_EXECUTABLE} fetch_externals.py -c
              ${ethos_u_base_tag}.json fetch
      WORKING_DIRECTORY ${ETHOS_SDK_PATH}
    )
  endif()
  # Patch core_software to remove unused projects.
  set(core_software_base_rev "26.02")
  patch_ethos_u_repo(
    "${ETHOS_SDK_PATH}/core_software" "${core_software_base_rev}"
    "${patch_dir}" "${ET_DIR_PATH}"
  )
  # Always patch the core_platform repo since this is fast enough. TODO:
  # examples/arm/ethos-u-setup/core_platform/0002-*.patch and 0003-*.patch are
  # transient bridges that guard Armv8-M-only MPU init and the Armv7-M-and-newer
  # HardFault handler so the Corstone-300 target source compiles for older
  # Cortex-M cores. Once the equivalent guards land upstream in
  # ethos-u/core_platform and ${core_platform_base_rev} is bumped past those
  # commits, delete the 0002 and 0003 patches.
  set(core_platform_base_rev "26.02")
  patch_ethos_u_repo(
    "${ETHOS_SDK_PATH}/core_platform" "${core_platform_base_rev}"
    "${patch_dir}" "${ET_DIR_PATH}"
  )
endfunction()

function(add_corstone_subdirectory SYSTEM_CONFIG ETHOS_SDK_PATH)
  if(SYSTEM_CONFIG MATCHES "Ethos_U55" OR SYSTEM_CONFIG MATCHES "Ethos_U65")
    add_subdirectory(
      ${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target
    )
  elseif(SYSTEM_CONFIG MATCHES "Ethos_U85")
    add_subdirectory(
      ${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target
    )
  else()
    message(FATAL_ERROR "Unsupported SYSTEM_CONFIG ${SYSTEM_CONFIG}.")
  endif()
endfunction()

function(configure_timing_adapters SYSTEM_CONFIG MEMORY_MODE)
  if(SYSTEM_CONFIG MATCHES "Ethos_U55_High_End_Embedded")
    set(TARGET_BOARD
        "corstone-300"
        PARENT_SCOPE
    )
    if(MEMORY_MODE MATCHES "Shared_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
    elseif(MEMORY_MODE MATCHES "Sram_Only")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # This is just example numbers and you should make this match
                  # your hardware SRAM
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
                  # Set the second Timing Adapter to SRAM latency & bandwidth
                  ETHOSU_TA_MAXR_1=8
                  ETHOSU_TA_MAXW_1=8
                  ETHOSU_TA_MAXRW_1=0
                  ETHOSU_TA_RLATENCY_1=32
                  ETHOSU_TA_WLATENCY_1=32
                  ETHOSU_TA_PULSE_ON_1=3999
                  ETHOSU_TA_PULSE_OFF_1=1
                  ETHOSU_TA_BWCAP_1=4000
                  ETHOSU_TA_PERFCTRL_1=0
                  ETHOSU_TA_PERFCNT_1=0
                  ETHOSU_TA_MODE_1=1
                  ETHOSU_TA_HISTBIN_1=0
                  ETHOSU_TA_HISTCNT_1=0
      )

    else()
      message(
        FATAL_ERROR
          "Unsupported memory_mode ${MEMORY_MODE} for the Ethos-U55. The Ethos-U55 supports only Shared_Sram and Sram_Only."
      )
    endif()
  elseif(SYSTEM_CONFIG MATCHES "Ethos_U55_Deep_Embedded")
    add_subdirectory(
      ${ETHOS_SDK_PATH}/core_platform/targets/corstone-300 target
    )
    set(TARGET_BOARD
        "corstone-300"
        PARENT_SCOPE
    )
    if(MEMORY_MODE MATCHES "Shared_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
    elseif(MEMORY_MODE MATCHES "Sram_Only")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
                  # Set the second Timing Adapter to SRAM latency & bandwidth
                  ETHOSU_TA_MAXR_1=4
                  ETHOSU_TA_MAXW_1=4
                  ETHOSU_TA_MAXRW_1=0
                  ETHOSU_TA_RLATENCY_1=8
                  ETHOSU_TA_WLATENCY_1=8
                  ETHOSU_TA_PULSE_ON_1=3999
                  ETHOSU_TA_PULSE_OFF_1=1
                  ETHOSU_TA_BWCAP_1=4000
                  ETHOSU_TA_PERFCTRL_1=0
                  ETHOSU_TA_PERFCNT_1=0
                  ETHOSU_TA_MODE_1=1
                  ETHOSU_TA_HISTBIN_1=0
                  ETHOSU_TA_HISTCNT_1=0
      )
    else()
      message(
        FATAL_ERROR
          "Unsupported memory_mode ${MEMORY_MODE} for the Ethos-U55. The Ethos-U55 supports only Shared_Sram and Sram_Only."
      )
    endif()
  elseif(SYSTEM_CONFIG STREQUAL "Ethos_U65_High_End")
    set(TARGET_BOARD
        "corstone-300"
        PARENT_SCOPE
    )
    if(MEMORY_MODE MATCHES "Shared_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
                  # SRAM
                  ETHOSU_TA_MAXR_0=16
                  ETHOSU_TA_MAXW_0=16
                  ETHOSU_TA_MAXRW_0=0
                  ETHOSU_TA_RLATENCY_0=32
                  ETHOSU_TA_WLATENCY_0=32
                  ETHOSU_TA_PULSE_ON_0=15999
                  ETHOSU_TA_PULSE_OFF_0=1
                  ETHOSU_TA_BWCAP_0=16000
                  ETHOSU_TA_PERFCTRL_0=0
                  ETHOSU_TA_PERFCNT_0=0
                  ETHOSU_TA_MODE_0=1
                  ETHOSU_TA_HISTBIN_0=0
                  ETHOSU_TA_HISTCNT_0=0
                  # DRAM
                  ETHOSU_TA_MAXR_1=24
                  ETHOSU_TA_MAXW_1=12
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
    elseif(MEMORY_MODE MATCHES "Sram_Only")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
                  # SRAM
                  ETHOSU_TA_MAXR_0=16
                  ETHOSU_TA_MAXW_0=16
                  ETHOSU_TA_MAXRW_0=0
                  ETHOSU_TA_RLATENCY_0=32
                  ETHOSU_TA_WLATENCY_0=32
                  ETHOSU_TA_PULSE_ON_0=15999
                  ETHOSU_TA_PULSE_OFF_0=1
                  ETHOSU_TA_BWCAP_0=16000
                  ETHOSU_TA_PERFCTRL_0=0
                  ETHOSU_TA_PERFCNT_0=0
                  ETHOSU_TA_MODE_0=1
                  ETHOSU_TA_HISTBIN_0=0
                  ETHOSU_TA_HISTCNT_0=0
                  # Set the second Timing Adapter to SRAM latency & bandwidth
                  ETHOSU_TA_MAXR_1=16
                  ETHOSU_TA_MAXW_1=16
                  ETHOSU_TA_MAXRW_1=0
                  ETHOSU_TA_RLATENCY_1=32
                  ETHOSU_TA_WLATENCY_1=32
                  ETHOSU_TA_PULSE_ON_1=15999
                  ETHOSU_TA_PULSE_OFF_1=1
                  ETHOSU_TA_BWCAP_1=16000
                  ETHOSU_TA_PERFCTRL_1=0
                  ETHOSU_TA_PERFCNT_1=0
                  ETHOSU_TA_MODE_1=1
                  ETHOSU_TA_HISTBIN_1=0
                  ETHOSU_TA_HISTCNT_1=0
      )
    elseif(MEMORY_MODE MATCHES "Dedicated_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
      message(
        FATAL_ERROR
          "Unsupported memory_mode ${MEMORY_MODE} for the Ethos-U65. The Ethos-U65 supports Shared_Sram and Sram_Only in this runner."
      )
    endif()
  elseif(SYSTEM_CONFIG MATCHES "Ethos_U85_SYS_DRAM_Low")
    add_subdirectory(
      ${ETHOS_SDK_PATH}/core_platform/targets/corstone-320 target
    )
    set(TARGET_BOARD
        "corstone-320"
        PARENT_SCOPE
    )
    if(MEMORY_MODE MATCHES "Dedicated_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
    elseif(MEMORY_MODE MATCHES "Sram_Only")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
                  # Set the second Timing Adapter to SRAM latency & bandwidth
                  ETHOSU_TA_MAXR_1=8
                  ETHOSU_TA_MAXW_1=8
                  ETHOSU_TA_MAXRW_1=0
                  ETHOSU_TA_RLATENCY_1=16
                  ETHOSU_TA_WLATENCY_1=16
                  ETHOSU_TA_PULSE_ON_1=3999
                  ETHOSU_TA_PULSE_OFF_1=1
                  ETHOSU_TA_BWCAP_1=4000
                  ETHOSU_TA_PERFCTRL_1=0
                  ETHOSU_TA_PERFCNT_1=0
                  ETHOSU_TA_MODE_1=1
                  ETHOSU_TA_HISTBIN_1=0
                  ETHOSU_TA_HISTCNT_1=0
      )
    endif()
  elseif(SYSTEM_CONFIG STREQUAL "Ethos_U85_SYS_DRAM_Mid"
         OR SYSTEM_CONFIG STREQUAL "Ethos_U85_SYS_DRAM_High"
  )
    set(TARGET_BOARD
        "corstone-320"
        PARENT_SCOPE
    )
    if(MEMORY_MODE MATCHES "Dedicated_Sram")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
    elseif(MEMORY_MODE MATCHES "Sram_Only")
      target_compile_definitions(
        ethosu_target_common
        INTERFACE # Configure NPU architecture timing adapters This is just
                  # example numbers and you should make this match your hardware
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
                  # Set the second Timing Adapter to SRAM latency & bandwidth
                  ETHOSU_TA_MAXR_1=8
                  ETHOSU_TA_MAXW_1=8
                  ETHOSU_TA_MAXRW_1=0
                  ETHOSU_TA_RLATENCY_1=32
                  ETHOSU_TA_WLATENCY_1=32
                  ETHOSU_TA_PULSE_ON_1=3999
                  ETHOSU_TA_PULSE_OFF_1=1
                  ETHOSU_TA_BWCAP_1=4000
                  ETHOSU_TA_PERFCTRL_1=0
                  ETHOSU_TA_PERFCNT_1=0
                  ETHOSU_TA_MODE_1=1
                  ETHOSU_TA_HISTBIN_1=0
                  ETHOSU_TA_HISTCNT_1=0
      )
    endif()
  else()
    message(FATAL_ERROR "Unsupported SYSTEM_CONFIG: ${SYSTEM_CONFIG}")
  endif()

  # The REGIONCFG registers of the Ethos-U control whether the NPU reads/writes
  # data through the SRAM or the external memory. By default, the Ethos-U driver
  # provides REGIONCFG configuration for Shared Sram memory mode. For Sram_Only
  # and Dedicated_Sram memory modes, we need to change the settings for optimal
  # performance.
  #
  # Currently, the convention used by Vela and the Ethos-U driver is that the
  # NPU uses: Region 0 for traffic of the Read-Only data(weights & biases)
  # Region 1 for traffic of of the intermediate Read/Write buffers required for
  # the computation Region 2 for traffic of of the cache in Dedicated_Sram
  # memory mode(not applicable in Sram_Only or Shared_Sram)
  #
  # NOTE: The above convention is determined by the Vela compiler and the
  # Ethos-U driver and can change in the future.
  #
  # Common definitions: For Ethos-U55/U65/U85, region configs are set as: 0 or 1
  # = AXI0 (Ethos-U55 or Ethos-U65) or AXI_SRAM(Ethos-U85) 2 or 3 = AXI1
  # (Ethos-U55 or Ethos-U65) or AXI_EXT(Ethos-U85)
  #
  # When we compile a model for Sram_Only, the memory traffic for Region 0 and
  # Region 1 should pass via the SRAM(hence regioncfg = 1) When we compile a
  # model for Dedicated_Sram, the memory traffic for Region 0 should pass via
  # the external memory(3), the memory traffic of Region 1 should pass via the
  # external memory(3) and the traffic for Region 2 should pass via the SRAM(0)
  #

  if(MEMORY_MODE MATCHES "Sram_Only")
    target_compile_definitions(
      ethosu_core_driver
      PRIVATE NPU_QCONFIG=1
              NPU_REGIONCFG_0=1
              NPU_REGIONCFG_1=0
              NPU_REGIONCFG_2=0
              NPU_REGIONCFG_3=0
              NPU_REGIONCFG_4=0
              NPU_REGIONCFG_5=0
              NPU_REGIONCFG_6=0
              NPU_REGIONCFG_7=0
    )
  elseif(MEMORY_MODE MATCHES "Dedicated_Sram")
    target_compile_definitions(
      ethosu_core_driver
      PRIVATE NPU_QCONFIG=3
              NPU_REGIONCFG_0=3
              NPU_REGIONCFG_1=3
              NPU_REGIONCFG_2=0
              NPU_REGIONCFG_3=0
              NPU_REGIONCFG_4=0
              NPU_REGIONCFG_5=0
              NPU_REGIONCFG_6=0
              NPU_REGIONCFG_7=0
    )
  endif()

endfunction()

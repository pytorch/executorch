# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

function(fetch_ethos_u_content ETHOS_SDK_PATH ET_DIR_PATH)
  message(STATUS "Fetching Ethos-U content into ${ETHOS_SDK_PATH}")

  file(MAKE_DIRECTORY ${ETHOS_SDK_PATH}/../ethos_u)
  include(FetchContent)
  set(ethos_u_base_tag "25.05")
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
  set(ethos_u_base_rev "24950bd4381b6c51db0349a229f8ba86b8e1093f")
  execute_process(
    COMMAND
      bash -c
      "pwd && source backends/arm/scripts/utils.sh && patch_repo ${ETHOS_SDK_PATH} ${ethos_u_base_rev} ${patch_dir}"
    WORKING_DIRECTORY ${ET_DIR_PATH} COMMAND_ECHO STDOUT
  )
  # Get ethos_u externals only if core_platform folder does not already exist.
  if(NOT EXISTS "${ETHOS_SDK_PATH}/core_platform")
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} fetch_externals.py -c
              ${ethos_u_base_tag}.json fetch
      WORKING_DIRECTORY ${ETHOS_SDK_PATH} COMMAND_ECHO STDOUT
    )
  endif()
  # Patch core_software to remove unused projects.
  set(core_software_base_rev "55904c3da73c876c6d6c58290938ae217a8b94bd")
  execute_process(
    COMMAND
      bash -c
      "pwd && source backends/arm/scripts/utils.sh && patch_repo ${ETHOS_SDK_PATH}/core_software ${core_software_base_rev} ${patch_dir}"
    WORKING_DIRECTORY ${ET_DIR_PATH} COMMAND_ECHO STDOUT
  )
  # Always patch the core_platform repo since this is fast enough.
  set(core_platform_base_rev "1916a9c984819c35b19c9e5c4c80d47e4e866420")
  execute_process(
    COMMAND
      bash -c
      "pwd && source backends/arm/scripts/utils.sh && patch_repo ${ETHOS_SDK_PATH}/core_platform ${core_platform_base_rev} ${patch_dir}"
    WORKING_DIRECTORY ${ET_DIR_PATH} COMMAND_ECHO STDOUT
  )
endfunction()

function(add_corstone_subdirectory SYSTEM_CONFIG ETHOS_SDK_PATH)
  if(SYSTEM_CONFIG MATCHES "Ethos_U55")
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
  if(MEMORY_MODE MATCHES "Dedicated_Sram")
    target_compile_definitions(
      ethosu_target_common INTERFACE ETHOSU_MODEL=1 ETHOSU_ARENA=1
    )
  elseif(MEMORY_MODE MATCHES "Shared_Sram" OR MEMORY_MODE MATCHES "Sram_Only")
    target_compile_definitions(
      ethosu_target_common INTERFACE ETHOSU_MODEL=1 ETHOSU_ARENA=0
    )
  else()
    message(
      FATAL_ERROR
        "Unsupported MEMORY_MODE ${MEMORY_MODE}. Memory_mode can be Shared_Sram, Sram_Only or Dedicated_Sram(applicable for the Ethos-U85)"
    )
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

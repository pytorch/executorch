# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

function(arm_ethos_u_content_ready SDK_PATH OUT_VAR)
  cmake_parse_arguments(
    ARG "DRIVER_ONLY" "SYSTEM_CONFIG;MISSING_UNMANAGED_PATH" "" ${ARGN}
  )
  get_filename_component(SDK_PATH "${SDK_PATH}" ABSOLUTE)
  set(_arm_ethos_managed_paths
      core_platform/targets/corstone-300/CMakeLists.txt
      core_platform/targets/corstone-320/CMakeLists.txt
      core_software/CMakeLists.txt
      core_software/core_driver/include/ethosu_driver.h
      core_software/cmsis_6/CMSIS/Core/Include/cmsis_compiler.h
      core_software/Cortex_DFP/ARM.Cortex_DFP.pdsc
      core_software/Cortex_DFP/Device
      core_software/cmsis-nn/Include/arm_nnfunctions.h
      core_software/cmsis-view/EventRecorder/Source/EventRecorder.c
  )
  list(TRANSFORM _arm_ethos_managed_paths PREPEND "${SDK_PATH}/")
  if(ARG_DRIVER_ONLY)
    set(_arm_ethos_required_paths
        "${SDK_PATH}/core_software/core_driver/include/ethosu_driver.h"
    )
  elseif(ARG_SYSTEM_CONFIG)
    if(ARG_SYSTEM_CONFIG MATCHES "Ethos_U55|Ethos_U65")
      set(_arm_ethos_platform corstone-300)
    elseif(ARG_SYSTEM_CONFIG MATCHES "Ethos_U85")
      set(_arm_ethos_platform corstone-320)
    else()
      message(FATAL_ERROR "Unsupported SYSTEM_CONFIG ${ARG_SYSTEM_CONFIG}.")
    endif()

    # Match core_software's defaults while preserving caller-supplied paths.
    if(NOT DEFINED ETHOSU_CORE_SOFTWARE_PATH)
      set(ETHOSU_CORE_SOFTWARE_PATH "${SDK_PATH}/core_software")
    endif()
    if(NOT DEFINED CORE_DRIVER_PATH)
      set(CORE_DRIVER_PATH "${ETHOSU_CORE_SOFTWARE_PATH}/core_driver")
    endif()
    if(NOT DEFINED CMSIS_VER)
      set(CMSIS_VER 6)
    endif()
    if(NOT DEFINED CMSIS_PATH)
      if(CMSIS_VER EQUAL 5)
        set(CMSIS_PATH "${ETHOSU_CORE_SOFTWARE_PATH}/cmsis")
      else()
        set(CMSIS_PATH "${ETHOSU_CORE_SOFTWARE_PATH}/cmsis_6")
      endif()
    endif()
    if(NOT DEFINED CMSIS_VIEW_PATH)
      set(CMSIS_VIEW_PATH "${ETHOSU_CORE_SOFTWARE_PATH}/cmsis-view")
    endif()
    if(NOT CMSIS_NN_LOCAL_PATH)
      set(CMSIS_NN_LOCAL_PATH "${SDK_PATH}/core_software/cmsis-nn")
    endif()
    set(_arm_ethos_required_paths
        "${SDK_PATH}/core_platform/targets/${_arm_ethos_platform}/CMakeLists.txt"
        "${ETHOSU_CORE_SOFTWARE_PATH}/CMakeLists.txt"
        "${CORE_DRIVER_PATH}/include/ethosu_driver.h"
        "${CMSIS_PATH}/CMSIS/Core/Include/cmsis_compiler.h"
        "${CMSIS_NN_LOCAL_PATH}/Include/arm_nnfunctions.h"
        "${CMSIS_VIEW_PATH}/EventRecorder/Source/EventRecorder.c"
    )
    if(CMSIS_VER EQUAL 5)
      list(APPEND _arm_ethos_required_paths "${CMSIS_PATH}/Device/ARM")
    else()
      if(NOT DEFINED CORTEX_DFP_PATH)
        set(CORTEX_DFP_PATH "${ETHOSU_CORE_SOFTWARE_PATH}/Cortex_DFP")
      endif()
      list(APPEND _arm_ethos_required_paths "${CORTEX_DFP_PATH}/Device")
    endif()
  else()
    # Provisioning checks the full manifest, regardless of the current target.
    set(_arm_ethos_required_paths ${_arm_ethos_managed_paths})
  endif()
  set(_arm_ethos_ready TRUE)
  set(_arm_ethos_missing_unmanaged_path "")
  foreach(_arm_ethos_path IN LISTS _arm_ethos_required_paths)
    get_filename_component(_arm_ethos_path "${_arm_ethos_path}" ABSOLUTE)
    if(NOT EXISTS "${_arm_ethos_path}")
      set(_arm_ethos_ready FALSE)
      if(NOT _arm_ethos_missing_unmanaged_path AND NOT _arm_ethos_path IN_LIST
                                                   _arm_ethos_managed_paths
      )
        set(_arm_ethos_missing_unmanaged_path "${_arm_ethos_path}")
      endif()
    endif()
  endforeach()
  if(ARG_MISSING_UNMANAGED_PATH)
    set(${ARG_MISSING_UNMANAGED_PATH}
        "${_arm_ethos_missing_unmanaged_path}"
        PARENT_SCOPE
    )
  endif()
  set(${OUT_VAR}
      ${_arm_ethos_ready}
      PARENT_SCOPE
  )
endfunction()

function(arm_ethos_u_default_fetch SDK_PATH OUT_VAR)
  arm_ethos_u_content_ready("${SDK_PATH}" _arm_ethos_ready ${ARGN})
  if(_arm_ethos_ready)
    set(${OUT_VAR}
        OFF
        PARENT_SCOPE
    )
  else()
    set(${OUT_VAR}
        ON
        PARENT_SCOPE
    )
  endif()
endfunction()

function(arm_ensure_ethos_u_content SDK_PATH EXECUTORCH_ROOT FETCH_REQUESTED)
  arm_ethos_u_content_ready(
    "${SDK_PATH}" _arm_ethos_ready_before MISSING_UNMANAGED_PATH
    _arm_ethos_missing_unmanaged_path ${ARGN}
  )

  if(_arm_ethos_ready_before)
    return()
  endif()

  if(_arm_ethos_missing_unmanaged_path)
    message(
      FATAL_ERROR
        "Cannot automatically repair missing Ethos-U dependency ${_arm_ethos_missing_unmanaged_path}. Fetching only populates the managed SDK at ${SDK_PATH}. Populate the dependency manually or remove the dependency path/CMSIS_VER overrides to use SDK defaults."
    )
  endif()

  if(NOT FETCH_REQUESTED)
    message(
      FATAL_ERROR
        "Missing or incomplete Ethos-U content at ${SDK_PATH}. Run examples/arm/setup.sh or enable FETCH_ETHOS_U_CONTENT=ON."
    )
  endif()

  fetch_ethos_u_content("${SDK_PATH}" "${EXECUTORCH_ROOT}")

  arm_ethos_u_content_ready("${SDK_PATH}" _arm_ethos_ready_after ${ARGN})
  if(NOT _arm_ethos_ready_after)
    message(
      FATAL_ERROR
        "Failed to fetch Ethos-U content into ${SDK_PATH}. Inspect the logs above."
    )
  endif()
endfunction()

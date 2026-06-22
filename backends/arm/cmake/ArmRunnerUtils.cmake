# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

# Helper routines shared by the standalone runner and any superbuild that reuses
# the runner targets.

function(arm_runner_require_baremetal_targets)
  if(NOT TARGET extension_runner_util)
    message(
      FATAL_ERROR
        "extension_runner_util target missing. Configure ExecuTorch (or the standalone runner) with EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON."
    )
  endif()

  if(NOT TARGET quantized_ops_lib OR NOT TARGET quantized_kernels)
    message(
      FATAL_ERROR
        "quantized kernels not found. Ensure EXECUTORCH_BUILD_KERNELS_QUANTIZED=ON when configuring ExecuTorch."
    )
  endif()

  if(NOT TARGET cortex_m_ops_lib OR NOT TARGET cortex_m_kernels)
    message(
      FATAL_ERROR
        "cortex_m backend not found. Ensure EXECUTORCH_BUILD_CORTEX_M=ON when configuring ExecuTorch."
    )
  endif()
endfunction()

# Ensure a runner target emits its binary to a predictable location. Uses
# FALLBACK_DIR when TARGET_NAME has no runtime output directory set, and also
# fills per-configuration runtime output directories for multi-config generators
# when they are unset.
function(arm_runner_configure_runtime_output TARGET_NAME FALLBACK_DIR)
  if(NOT TARGET ${TARGET_NAME})
    return()
  endif()

  get_target_property(_base_runtime_dir ${TARGET_NAME} RUNTIME_OUTPUT_DIRECTORY)
  if(NOT _base_runtime_dir
     OR _base_runtime_dir STREQUAL "_base_runtime_dir-NOTFOUND"
     OR "${_base_runtime_dir}" STREQUAL ""
  )
    set_target_properties(
      ${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${FALLBACK_DIR}"
    )
    set(_base_runtime_dir "${FALLBACK_DIR}")
  endif()

  if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${_cfg} _cfg_upper)
      set(_cfg_prop "RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper}")
      get_target_property(_cfg_dir ${TARGET_NAME} ${_cfg_prop})
      if(NOT _cfg_dir
         OR _cfg_dir STREQUAL "_cfg_dir-NOTFOUND"
         OR "${_cfg_dir}" STREQUAL ""
      )
        set_target_properties(
          ${TARGET_NAME} PROPERTIES ${_cfg_prop} "${_base_runtime_dir}/${_cfg}"
        )
      endif()
    endforeach()
  endif()
endfunction()

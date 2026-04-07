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

# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include_guard(GLOBAL)

function(arm_ethos_u_content_ready SDK_PATH OUT_VAR)
  if(EXISTS "${SDK_PATH}/core_platform" AND EXISTS "${SDK_PATH}/core_software")
    set(${OUT_VAR}
        TRUE
        PARENT_SCOPE
    )
  else()
    set(${OUT_VAR}
        FALSE
        PARENT_SCOPE
    )
  endif()
endfunction()

function(arm_ethos_u_default_fetch SDK_PATH OUT_VAR)
  arm_ethos_u_content_ready("${SDK_PATH}" _arm_ethos_ready)
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
  arm_ethos_u_content_ready("${SDK_PATH}" _arm_ethos_ready_before)

  if(_arm_ethos_ready_before)
    return()
  endif()

  if(NOT FETCH_REQUESTED)
    message(
      FATAL_ERROR
        "No Ethos-U content found at ${SDK_PATH}. Run examples/arm/setup.sh or enable FETCH_ETHOS_U_CONTENT=ON."
    )
  endif()

  fetch_ethos_u_content(${SDK_PATH} ${EXECUTORCH_ROOT})

  arm_ethos_u_content_ready("${SDK_PATH}" _arm_ethos_ready_after)
  if(NOT _arm_ethos_ready_after)
    message(
      FATAL_ERROR
        "Failed to fetch Ethos-U content into ${SDK_PATH}. Inspect the logs above."
    )
  endif()
endfunction()

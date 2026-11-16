# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Locates the eIQ Neutron SDK and the selected target. Sets:
# EIQ_NEUTRON_TARGET_DIR, EIQ_NEUTRON_TARGET_INCLUDE_DIR.

if(NOT DEFINED EIQ_NEUTRON_TARGET)
  set(EIQ_NEUTRON_TARGET
      "imxrt700"
      CACHE STRING "Neutron Target to build against"
  )
endif()

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED
)

message(STATUS "Looking for eIQ Neutron SDK")
execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "import eiq_neutron_sdk, os; print(os.path.dirname(eiq_neutron_sdk.__file__))"
  OUTPUT_VARIABLE EIQ_NEUTRON_SDK_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Looking for eIQ Neutron SDK -- found: ${EIQ_NEUTRON_SDK_PATH}")

message(STATUS "Looking for Neutron target ${EIQ_NEUTRON_TARGET}")
set(EIQ_NEUTRON_TARGET_DIR
    "${EIQ_NEUTRON_SDK_PATH}/target/${EIQ_NEUTRON_TARGET}"
)
if(NOT EXISTS ${EIQ_NEUTRON_TARGET_DIR})
  message(
    FATAL_ERROR "Looking for Neutron target ${EIQ_NEUTRON_TARGET} -- not found"
  )
endif()
message(
  STATUS
    "Looking for Neutron target ${EIQ_NEUTRON_TARGET} -- found: ${EIQ_NEUTRON_TARGET_DIR}"
)

set(EIQ_NEUTRON_TARGET_INCLUDE_DIR "${EIQ_NEUTRON_TARGET_DIR}/common/include"
                                   "${EIQ_NEUTRON_TARGET_DIR}/driver/include"
)

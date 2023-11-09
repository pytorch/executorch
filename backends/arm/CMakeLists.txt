# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)
project(arm_backend)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build config requirements for Arm Backend
if(NOT EXECUTORCH_BUILD_ARM_BAREMETAL)
  message(FATAL_ERROR, "expected EXECUTORCH_BUILD_ARM_BAREMETAL")
endif()
if(NOT DEFINED ETHOSU_DDK_PATH)
    message(FATAL_ERROR "ETHOSU_DDK_PATH must be set and point to ethos-u-core-driver")
endif()

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

# Third-party include for Ethos-U driver
set(DRIVER_ETHOSU_INCLUDE_DIR "${ETHOSU_DDK_PATH}/include")
include_directories( ${DRIVER_ETHOSU_INCLUDE_DIR} )

set(_arm_baremetal_sources
  backends/arm/runtime/ArmBackendEthosU.cpp
  backends/arm/runtime/VelaBinStream.cpp
)
list(TRANSFORM _arm_baremetal_sources PREPEND "${EXECUTORCH_ROOT}/")

add_library(
  executorch_delegate_ethos_u
  STATIC ${_arm_baremetal_sources}
)
target_include_directories(
  executorch_delegate_ethos_u
  PUBLIC
  ${_common_include_directories}
)
target_include_directories(
  executorch_delegate_ethos_u
  PUBLIC
  ${DRIVER_ETHOSU_INCLUDE_DIR}
)

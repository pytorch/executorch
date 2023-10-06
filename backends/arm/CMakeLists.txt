# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.19)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source root directory for executorch.
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)

set(_common_include_directories ${EXECUTORCH_ROOT}/..)

include(cmake/Dependencies.cmake)

set(_arm_baremetal_sources backends/arm/runtime/ArmBackendEthosU.cpp)
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

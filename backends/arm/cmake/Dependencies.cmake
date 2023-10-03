# Copyright 2023 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")

# Ethos-U driver
set(DRIVER_ETHOSU_SOURCE_DIR "${THIRD_PARTY_ROOT}/ethos-u-core-driver")
set(DRIVER_ETHOSU_INCLUDE_DIR "${THIRD_PARTY_ROOT}/ethos-u-core-driver/include")
add_subdirectory( ${DRIVER_ETHOSU_SOURCE_DIR} )
include_directories( ${DRIVER_ETHOSU_INCLUDE_DIR} )

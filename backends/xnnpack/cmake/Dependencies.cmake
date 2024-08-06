# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The logic is copied from
# https://github.com/pytorch/pytorch/blob/main/cmake/Dependencies.cmake
set(THIRD_PARTY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third-party")

# --- XNNPACK

# Setting this global PIC flag for all XNNPACK targets. This is needed for
# Object libraries within XNNPACK which must be PIC to successfully link this
# static libXNNPACK
set(ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG
    ${CMAKE_POSITION_INDEPENDENT_CODE}
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(XNNPACK_SOURCE_DIR "${THIRD_PARTY_ROOT}/XNNPACK")
set(XNNPACK_INCLUDE_DIR "${XNNPACK_SOURCE_DIR}/include")
set(XNNPACK_LIBRARY_TYPE
    "static"
    CACHE STRING ""
)
set(XNNPACK_BUILD_BENCHMARKS
    OFF
    CACHE BOOL ""
)
set(XNNPACK_BUILD_TESTS
    OFF
    CACHE BOOL ""
)
set(XNNPACK_ENABLE_AVXVNNI
    OFF
    CACHE BOOL ""
)
set(XNNPACK_ENABLE_KLEIDIAI
    OFF
    CACHE BOOL ""
)
add_subdirectory("${XNNPACK_SOURCE_DIR}")
include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
list(APPEND xnnpack_third_party XNNPACK)

# Revert PIC Flag to what it originally was
set(CMAKE_POSITION_INDEPENDENT_CODE
    ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
)

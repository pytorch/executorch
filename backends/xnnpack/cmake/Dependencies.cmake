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
# Work around observed failure:
# https://github.com/pytorch/executorch/pull/10362#issuecomment-2906391232
set(XNNPACK_ENABLE_AVX512VNNIGFNI
    OFF
    CACHE BOOL ""
)
# Enable ARM SME2 by default. Can be disabled with -DXNNPACK_ENABLE_ARM_SME2=OFF
# for Android emulator compatibility, which may crash with SME2 kernels.
set(XNNPACK_ENABLE_ARM_SME2
    ON
    CACHE BOOL ""
)
if(EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  set(XNNPACK_ENABLE_KLEIDIAI
      ON
      CACHE BOOL ""
  )
else()
  set(XNNPACK_ENABLE_KLEIDIAI
      OFF
      CACHE BOOL ""
  )
endif()

if(WIN32)
  # These XNNPACK options don't currently build on Windows.
  set_overridable_option(XNNPACK_ENABLE_AVX256SKX OFF)
  set_overridable_option(XNNPACK_ENABLE_AVX256VNNI OFF)
  set_overridable_option(XNNPACK_ENABLE_AVX256VNNIGFNI OFF)
  set_overridable_option(XNNPACK_ENABLE_AVX512BF16 OFF)
endif()

set(XNNPACK_BUILD_ALL_MICROKERNELS
    OFF
    CACHE BOOL ""
)
add_subdirectory("${XNNPACK_SOURCE_DIR}")
include_directories(SYSTEM ${XNNPACK_INCLUDE_DIR})
list(APPEND xnnpack_third_party XNNPACK)
install(
  TARGETS xnnpack-microkernels-prod
  EXPORT ExecuTorchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(EXECUTORCH_XNNPACK_ENABLE_KLEIDI)
  if(TARGET kleidiai)
    install(
      TARGETS kleidiai
      EXPORT ExecuTorchTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
  endif()
endif()

# Revert PIC Flag to what it originally was
set(CMAKE_POSITION_INDEPENDENT_CODE
    ${ORIGINAL_CMAKE_POSITION_INDEPENDENT_CODE_FLAG}
)

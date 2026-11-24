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
  # XNNPACK downloads KleidiAI with ExternalProject, which never
  # retries: a transient 502 from gitlab.arm.com fails the whole build
  # (seen repeatedly in test-models-linux-aarch64). Download it here
  # with retries and point XNNPACK at the local copy instead (XNNPACK
  # skips its own download when KLEIDIAI_SOURCE_DIR is defined). The
  # version and hash must match XNNPACK's cmake/DownloadKleidiAI.cmake.
  if(NOT DEFINED KLEIDIAI_SOURCE_DIR
     AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(KLEIDIAI_VERSION "b87ef9c94f45f11c81a6b1fdaed1b2b45ea58c0c")
    set(KLEIDIAI_URL
        "https://gitlab.arm.com/kleidi/kleidiai/-/archive/${KLEIDIAI_VERSION}/kleidiai-${KLEIDIAI_VERSION}.zip"
    )
    set(KLEIDIAI_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/kleidiai-download")
    set(KLEIDIAI_ZIP "${KLEIDIAI_DOWNLOAD_DIR}/kleidiai.zip")
    file(MAKE_DIRECTORY "${KLEIDIAI_DOWNLOAD_DIR}")
    set(KLEIDIAI_DOWNLOADED FALSE)
    foreach(KLEIDIAI_ATTEMPT RANGE 1 5)
      file(
        DOWNLOAD "${KLEIDIAI_URL}" "${KLEIDIAI_ZIP}"
        EXPECTED_HASH
          SHA256=b147799b94c51f5e57492930bfd9e5294fb7ffe44fee1dbcd3f8048adeedd5e3
        STATUS KLEIDIAI_STATUS
        TIMEOUT 180
      )
      list(GET KLEIDIAI_STATUS 0 KLEIDIAI_CODE)
      if(KLEIDIAI_CODE EQUAL 0)
        set(KLEIDIAI_DOWNLOADED TRUE)
        break()
      endif()
      message(
        STATUS
          "KleidiAI download attempt ${KLEIDIAI_ATTEMPT} failed (${KLEIDIAI_STATUS}); retrying..."
      )
      execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 10)
    endforeach()
    if(NOT KLEIDIAI_DOWNLOADED)
      message(
        FATAL_ERROR
          "Failed to download KleidiAI from ${KLEIDIAI_URL} after 5 attempts: ${KLEIDIAI_STATUS}"
      )
    endif()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E tar xzf "${KLEIDIAI_ZIP}"
      WORKING_DIRECTORY "${KLEIDIAI_DOWNLOAD_DIR}"
    )
    set(KLEIDIAI_SOURCE_DIR
        "${KLEIDIAI_DOWNLOAD_DIR}/kleidiai-${KLEIDIAI_VERSION}"
        CACHE PATH "KleidiAI source directory (pre-downloaded by ExecuTorch)"
    )
  endif()
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
  # clang-cl (reported by CMake as Clang, not MSVC, so XNNPACK's own MSVC gate
  # does not catch it) crashes with an internal codegen error compiling some of
  # the AVX512-FP16 micro-kernels, so disable them on Windows.
  set_overridable_option(XNNPACK_ENABLE_AVX512FP16 OFF)
endif()

set(XNNPACK_BUILD_ALL_MICROKERNELS
    OFF
    CACHE BOOL ""
)

# XNNPACK nests variadic macros (XNN_RETURN_IF_ERROR expands to xnn_log_error),
# which MSVC's default legacy preprocessor mis-expands. Only observable where
# XNN_LOG_LEVEL is non-zero, i.e. Debug configurations. Not applied to clang-cl,
# which sets MSVC but already has a conforming preprocessor.
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/Zc:preprocessor)
endif()

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

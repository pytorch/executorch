# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Minimum version requirement for using this toolchain file. Do not call
# cmake_minimum_required() here, as that can reset policies for the parent
# project; instead, guard on CMAKE_VERSION explicitly.
if(CMAKE_VERSION VERSION_LESS 3.20)
  message(FATAL_ERROR "This toolchain file requires at least CMake 3.20")
endif()

# Toolchain root for the standalone aarch64-linux-musl cross compiler.
set(MUSL_TOOLCHAIN_ROOT
    ""
    CACHE PATH "Root of the aarch64-linux-musl toolchain"
)
if(MUSL_TOOLCHAIN_ROOT STREQUAL "" AND DEFINED ENV{MUSL_TOOLCHAIN_ROOT})
  set(MUSL_TOOLCHAIN_ROOT "$ENV{MUSL_TOOLCHAIN_ROOT}")
endif()
if(MUSL_TOOLCHAIN_ROOT STREQUAL "")
  message(
    FATAL_ERROR
      "MUSL_TOOLCHAIN_ROOT is required (e.g. -DMUSL_TOOLCHAIN_ROOT=/path/to/aarch64-linux-musl-cross or export MUSL_TOOLCHAIN_ROOT=...)"
  )
endif()

# Ensure the toolchain root is forwarded to try_compile checks.
set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES MUSL_TOOLCHAIN_ROOT)
set(_MUSL_SYSROOT "${MUSL_TOOLCHAIN_ROOT}/aarch64-linux-musl")
set(_MUSL_BIN_DIR "${MUSL_TOOLCHAIN_ROOT}/bin")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_SYSROOT
    "${_MUSL_SYSROOT}"
    CACHE PATH "Musl target sysroot"
)

set(CMAKE_C_COMPILER
    "${_MUSL_BIN_DIR}/aarch64-linux-musl-gcc"
    CACHE FILEPATH "Musl cross C compiler"
)
set(CMAKE_CXX_COMPILER
    "${_MUSL_BIN_DIR}/aarch64-linux-musl-g++"
    CACHE FILEPATH "Musl cross C++ compiler"
)
set(CMAKE_AR
    "${_MUSL_BIN_DIR}/aarch64-linux-musl-ar"
    CACHE FILEPATH "Musl archiver"
)
set(CMAKE_RANLIB
    "${_MUSL_BIN_DIR}/aarch64-linux-musl-ranlib"
    CACHE FILEPATH "Musl ranlib"
)
set(CMAKE_STRIP
    "${_MUSL_BIN_DIR}/aarch64-linux-musl-strip"
    CACHE FILEPATH "Musl strip"
)

set(CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

if(DEFINED ENV{PKG_CONFIG_SYSROOT_DIR})
  set(ENV{PKG_CONFIG_SYSROOT_DIR} $ENV{PKG_CONFIG_SYSROOT_DIR})
else()
  set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
endif()

if(DEFINED ENV{PKG_CONFIG_PATH})
  set(ENV{PKG_CONFIG_PATH}
      "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig:$ENV{PKG_CONFIG_PATH}"
  )
else()
  set(ENV{PKG_CONFIG_PATH}
      "${CMAKE_SYSROOT}/usr/lib/pkgconfig:${CMAKE_SYSROOT}/usr/share/pkgconfig"
  )
endif()

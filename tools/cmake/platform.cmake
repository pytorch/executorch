# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/common.cmake)

pragma_once()

# MARK: - Define platform

set(_supported_os)
macro(_define_platform_os NAME)
  string(TOUPPER "${NAME}" NAME_UPPER)
  string(TOLOWER "${NAME}" NAME_LOWER)
  set("PLATFORM_OS_${NAME_UPPER}" "${NAME_LOWER}")
  list(APPEND _supported_os "${NAME_LOWER}")
endmacro()

_define_platform_os("macosx")
_define_platform_os("linux")
_define_platform_os("ios")
_define_platform_os("ios_simulator")
_define_platform_os("android")

set(_supported_arch)
macro(_define_platform_arch NAME)
  string(TOUPPER "${NAME}" NAME_UPPER)
  string(TOLOWER "${NAME}" NAME_LOWER)
  set("PLATFORM_ARCH_${NAME_UPPER}" "${NAME_LOWER}")
  list(APPEND _supported_arch "${NAME_LOWER}")
endmacro()

_define_platform_arch("arm64")
_define_platform_arch("x86_64")

# MARK: - Host platform

if(APPLE AND CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(PLATFORM_HOST_OS ${PLATFORM_OS_MACOSX})
elseif(UNIX)
  set(PLATFORM_HOST_OS ${PLATFORM_OS_LINUX})
else()
  message(FATAL_ERROR "Unsupported host platform")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(PLATFORM_HOST_ARCH ${PLATFORM_ARCH_ARM64})
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(PLATFORM_HOST_ARCH ${PLATFORM_ARCH_X86_64})
else()
  message(FATAL_ERROR "Unsupported host architecture")
endif()

if(NOT PLATFORM_TARGET_OS)
  set(PLATFORM_TARGET_OS ${PLATFORM_HOST_OS})
elseif(NOT PLATFORM_TARGET_OS IN_LIST _supported_os)
  message(FATAL_ERROR "Unsupported PLATFORM_TARGET_OS: ${PLATFORM_TARGET_OS}. Choices: ${_supported_os}")
endif()
announce_configured_options(PLATFORM_TARGET_OS)

if(NOT PLATFORM_TARGET_ARCH)
  set(PLATFORM_TARGET_ARCH ${PLATFORM_HOST_ARCH})
elseif(NOT PLATFORM_TARGET_ARCH IN_LIST _supported_arch)
  message(FATAL_ERROR "Unsupported PLATFORM_TARGET_ARCH: ${PLATFORM_TARGET_ARCH}. Choices: ${_supported_arch}")
endif()
announce_configured_options(PLATFORM_TARGET_ARCH)

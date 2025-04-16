# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MARK: - Define platform

set(EXECUTORCH_SUPPORTED_PLATFORM_OS "")
macro(_define_platform_os NAME)
  string(TOUPPER "EXECUTORCH_PLATFORM_OS_${NAME}" NAME_UPPER)
  set(${NAME_UPPER} ${NAME})
  list(APPEND EXECUTORCH_SUPPORTED_PLATFORM_OS ${${NAME_UPPER}})
endmacro()

# Desktop OS
_define_platform_os("macos")
_define_platform_os("linux")
_define_platform_os("windows")
# Mobile OS
_define_platform_os("ios")
_define_platform_os("ios_simulator")
_define_platform_os("android")

set(EXECUTORCH_SUPPORTED_PLATFORM_ARCH "")
macro(_define_platform_arch NAME)
  string(TOUPPER "EXECUTORCH_PLATFORM_ARCH_${NAME}" NAME_UPPER)
  set(${NAME_UPPER} ${NAME})
  list(APPEND EXECUTORCH_SUPPORTED_PLATFORM_ARCH ${${NAME_UPPER}})
endmacro()

_define_platform_arch("arm64")
_define_platform_arch("x86_64")

# MARK: - Host platform

if(NOT EXECUTORCH_PLATFORM_HOST_OS)
  macro(_set_executorch_platform_host_os OS)
    set(EXECUTORCH_PLATFORM_HOST_OS ${OS} CACHE INTERNAL "Host operating system" FORCE)
  endmacro()

  if(APPLE AND CMAKE_SYSTEM_NAME MATCHES "Darwin")
    _set_executorch_platform_host_os(${EXECUTORCH_PLATFORM_OS_MACOS})
  elseif(UNIX)
    _set_executorch_platform_host_os(${EXECUTORCH_PLATFORM_OS_LINUX})
  elseif(WIN32)
    _set_executorch_platform_host_os(${EXECUTORCH_PLATFORM_OS_WINDOWS})
  else()
    message(FATAL_ERROR "Unsupported host operating system")
  endif()
endif()
announce_configured_options(EXECUTORCH_PLATFORM_HOST_OS)

if(NOT EXECUTORCH_PLATFORM_HOST_ARCH)
  macro(_set_executorch_platform_host_arch ARCH)
    set(EXECUTORCH_PLATFORM_HOST_ARCH ${ARCH} CACHE INTERNAL "Host architecture" FORCE)
  endmacro()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    _set_executorch_platform_host_arch(${EXECUTORCH_PLATFORM_ARCH_ARM64})
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    _set_executorch_platform_host_arch(${EXECUTORCH_PLATFORM_ARCH_X86_64})
  else()
    message(FATAL_ERROR "Unsupported host architecture")
  endif()
endif()
announce_configured_options(EXECUTORCH_PLATFORM_HOST_ARCH)

if(NOT EXECUTORCH_PLATFORM_TARGET_OS)
  # If a target os is not provided, build for the host.
  set(EXECUTORCH_PLATFORM_TARGET_OS ${EXECUTORCH_PLATFORM_HOST_OS} CACHE INTERNAL "Target operating system")
elseif(NOT EXECUTORCH_PLATFORM_TARGET_OS IN_LIST EXECUTORCH_SUPPORTED_PLATFORM_OS)
  message(FATAL_ERROR "Unsupported EXECUTORCH_PLATFORM_TARGET_OS: ${EXECUTORCH_PLATFORM_TARGET_OS}. Choices: ${EXECUTORCH_SUPPORTED_PLATFORM_OS}")
endif()
announce_configured_options(EXECUTORCH_PLATFORM_TARGET_OS)

if(NOT EXECUTORCH_PLATFORM_TARGET_ARCH)
  # If a target arch is not provided, build for the host.
  set(EXECUTORCH_PLATFORM_TARGET_ARCH ${EXECUTORCH_PLATFORM_HOST_ARCH} CACHE INTERNAL "Target architecture")
elseif(NOT EXECUTORCH_PLATFORM_TARGET_ARCH IN_LIST EXECUTORCH_SUPPORTED_PLATFORM_ARCH)
  message(FATAL_ERROR "Unsupported EXECUTORCH_PLATFORM_TARGET_ARCH: ${EXECUTORCH_PLATFORM_TARGET_ARCH}. Choices: ${EXECUTORCH_SUPPORTED_PLATFORM_ARCH}")
endif()
announce_configured_options(EXECUTORCH_PLATFORM_TARGET_ARCH)

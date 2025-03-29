# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

pragma_once()

if(NOT PLATFORM_TARGET_OS STREQUAL PLATFORM_OS_IOS)
  return()
endif()

if(NOT PLATFORM_TARGET_ARCH STREQUAL ${PLATFORM_ARCH_ARM64})
  message(FATAL_ERROR "Unsupported architecture for iOS: ${PLATFORM_TARGET_ARCH}")
endif()

find_program(XCODEBUILD_EXECUTABLE xcodebuild)
if(NOT XCODEBUILD_EXECUTABLE)
  message(FATAL_ERROR "xcodebuild not found. Please install either the standalone commandline tools or Xcode.")
endif()

set(IOS_SDK_NAME iphoneos CACHE INTERNAL "")

execute_process(
  COMMAND ${XCODEBUILD_EXECUTABLE} -version -sdk ${IOS_SDK_NAME} Path
  OUTPUT_VARIABLE IOS_SDK_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(IOS_SDK_PATH ${IOS_SDK_PATH} CACHE INTERNAL "")
announce_configured_options(IOS_SDK_PATH)

set(IOS_DEPLOYMENT_TARGET "13.0" CACHE INTERNAL "")
announce_configured_options(IOS_DEPLOYMENT_TARGET)

set(IOS_TARGET_TRIPLE_INT "${PLATFORM_TARGET_ARCH}-apple-${PLATFORM_TARGET_OS}${IOS_DEPLOYMENT_TARGET}" CACHE INTERNAL "")
announce_configured_options(IOS_TARGET_TRIPLE_INT)

add_library(ios_toolchain INTERFACE)
target_link_libraries(ios_toolchain INTERFACE cxx_toolchain)

target_compile_options(ios_toolchain
  INTERFACE
    -arch ${PLATFORM_TARGET_ARCH}
    -target ${IOS_TARGET_TRIPLE_INT}
    -isysroot ${IOS_SDK_PATH}
    -fembed-bitcode
    -fobjc-arc
    -fvisibility=hidden
)

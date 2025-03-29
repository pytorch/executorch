# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

pragma_once()

if(NOT PLATFORM_TARGET_OS STREQUAL PLATFORM_OS_MACOSX)
  return()
endif()

set(MACOS_SDK_NAME macosx CACHE INTERNAL "")

execute_process(
  COMMAND xcrun -sdk ${MACOS_SDK_NAME} --show-sdk-path
  OUTPUT_VARIABLE MACOS_SDK_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(MACOS_SDK_PATH ${MACOS_SDK_PATH} CACHE INTERNAL "")
announce_configured_options(MACOS_SDK_PATH)

set(MACOS_DEPLOYMENT_TARGET "11.0" CACHE INTERNAL "")
announce_configured_options(MACOS_DEPLOYMENT_TARGET)

set(MACOS_TARGET_TRIPLE_INT "${PLATFORM_TARGET_ARCH}-apple-${PLATFORM_TARGET_OS}${MACOS_DEPLOYMENT_TARGET}" CACHE INTERNAL "")
announce_configured_options(MACOS_TARGET_TRIPLE_INT)

add_library(macos_toolchain INTERFACE)
target_link_libraries(macos_toolchain INTERFACE cxx_toolchain)

target_compile_options(macos_toolchain
  INTERFACE
    -arch ${PLATFORM_TARGET_ARCH}
    -target ${MACOS_TARGET_TRIPLE_INT}
    -isysroot ${MACOS_SDK_PATH}
    -fembed-bitcode
    -fobjc-arc
    -fvisibility=hidden
)

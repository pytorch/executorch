# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(${PROJECT_SOURCE_DIR}/tools/cmake/platform.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_toolchain.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/ios_toolchain.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/macos_toolchain.cmake)

add_subdirectory(third-party)
add_subdirectory(schema)
add_subdirectory(hello)

print_configured_options()

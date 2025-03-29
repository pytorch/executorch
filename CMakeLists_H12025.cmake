# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_toolchain.cmake)

configure_cxx_toolchain(
  NAME common_cxx_toolchain
  CXX_STD 17
)

add_subdirectory(third-party)
add_subdirectory(schema)
add_subdirectory(hello)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_library.cmake)
include(${PROJECT_SOURCE_DIR}/tools/cmake/apple_library.cmake)

cxx_library(
  NAME hello.message
  SRCS message.cpp
  HEADERS message.h
  DEPS flatbuffers schema.program
)

apple_library(
  NAME hello.bonjour
  SRCS bonjour.mm
  IOS
  MACOS
  DEPS hello.message
  FRAMEWORKS Foundation
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/cxx_library.cmake)

cxx_library(
  NAME message
  SRCS message.cpp
  HEADERS message.h
  DEPS flatbuffers schema.program
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

include(${PROJECT_SOURCE_DIR}/tools/cmake/flatbuffer_cxx_library.cmake)

flatbuffer_cxx_library(
  NAME schema.scalar
  SCHEMAS scalar_type.fbs
)

flatbuffer_cxx_library(
  NAME schema.program
  SCHEMAS program.fbs
  DEPS schema.scalar
)

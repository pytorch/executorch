# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

et_cxx_test(
  graph_types_test SOURCES graph_types_test.cpp EXTRA_LIBS program_schema
)

target_include_directories(
  graph_types_test
  PRIVATE "${EXECUTORCH_ROOT}" "${CMAKE_INSTALL_PREFIX}/schema/include"
          "${EXECUTORCH_ROOT}/third-party/flatbuffers/include"
)

set_property(
  TEST graph_types_test
  PROPERTY
    ENVIRONMENT
    "ET_NATIVE_TEST_DATA=${CMAKE_CURRENT_SOURCE_DIR}/testdata/linear_4x4.bin"
)

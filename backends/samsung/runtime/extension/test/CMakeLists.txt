# Copyright (c) 2025 Samsung Electronics Co. LTD
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# logging

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs exynos_file_data_loader_test.cpp)

et_cxx_test(
  exynos_file_data_loader_test SOURCES ${_test_srcs} EXTRA_LIBS enn_backend
  enn_logging
)

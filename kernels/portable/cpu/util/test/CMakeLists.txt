# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)
include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

set(_test_srcs broadcast_indexes_range_test.cpp broadcast_test.cpp
               reduce_test.cpp vectorized_math_test.cpp
)

et_cxx_test(
  kernels_portable_cpu_util_test SOURCES ${_test_srcs} EXTRA_LIBS
  portable_kernels portable_ops_lib
)

find_package_torch_headers()
target_include_directories(
  kernels_portable_cpu_util_test PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_compile_definitions(
  kernels_portable_cpu_util_test PRIVATE ET_USE_PYTORCH_HEADERS
)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file should be formatted with
# ~~~
# cmake-format -i CMakeLists.txt
# ~~~
# It should also be cmake-lint clean.
#

cmake_minimum_required(VERSION 3.19)
project(backends_xnnpack_test)

# Use C++17 for test.
set(CMAKE_CXX_STANDARD 17)

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/build/Test.cmake)

set(_test_srcs # We can't put runtime/test_runtime_utils.cpp because we don't
               # build aten
    runtime/test_xnnexecutor.cpp
    ${EXECUTORCH_ROOT}/extension/threadpool/threadpool.cpp
    ${EXECUTORCH_ROOT}/extension/threadpool/threadpool_guard.cpp
    ${EXECUTORCH_ROOT}/extension/threadpool/test/threadpool_test.cpp
)

et_cxx_test(
  backends_xnnpack_test
  SOURCES
  ${_test_srcs}
  EXTRA_LIBS
  extension_threadpool
  xnnpack_backend
  XNNPACK
  pthreadpool
  cpuinfo
)
target_include_directories(
  backends_xnnpack_test
  PRIVATE ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/src
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
)

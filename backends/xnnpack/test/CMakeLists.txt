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

set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

include(${EXECUTORCH_ROOT}/tools/cmake/Test.cmake)

set(_test_srcs runtime/test_xnnexecutor.cpp
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
  xnnpack-microkernels-prod
)
target_include_directories(
  backends_xnnpack_test
  PRIVATE ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/src
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
)

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

set(_test_srcs runtime/test_xnnexecutor.cpp runtime/test_workspace_manager.cpp
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

# Graph runtime unit tests.
set(_graph_runtime_test_srcs
    runtime/test_graph_builder.cpp
    runtime/test_partition.cpp
    runtime/test_schedule.cpp
    runtime/test_shape_env.cpp
    runtime/test_quant_params.cpp
)

et_cxx_test(
  backends_xnnpack_graph_runtime_test
  SOURCES
  ${_graph_runtime_test_srcs}
  EXTRA_LIBS
  xnnpack_backend
  XNNPACK
  pthreadpool
  cpuinfo
  xnnpack-microkernels-prod
)
target_include_directories(
  backends_xnnpack_graph_runtime_test
  PRIVATE ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
)

# Graph runtime E2E tests (requires XNNPACK runtime).
et_cxx_test(
  backends_xnnpack_graph_e2e_test
  SOURCES
  runtime/test_e2e.cpp
  EXTRA_LIBS
  xnnpack_backend
  XNNPACK
  pthreadpool
  cpuinfo
  xnnpack-microkernels-prod
)
target_include_directories(
  backends_xnnpack_graph_e2e_test
  PRIVATE ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/XNNPACK/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/cpuinfo/include
          ${EXECUTORCH_ROOT}/backends/xnnpack/third-party/pthreadpool/include
)

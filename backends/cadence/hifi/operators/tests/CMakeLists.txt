# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# gtest op-level tests for the Cadence HiFi backend, cross-compiled for Xtensa
# and run on the Instruction Set Simulator (xt-run). Each test calls the kernel
# directly (impl::HiFi::native::<op>) and checks with EXPECT_TENSOR_EQ, mirroring
# the BUCK tests. Built only when EXECUTORCH_BUILD_CADENCE_OP_TESTS is ON; see
# .ci/scripts/test-cadence-xtensa.sh.

cmake_minimum_required(VERSION 3.19)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# Build googletest FOR the Xtensa target. find_package(GTest) would resolve the
# HOST gtest, which cannot link Xtensa objects, so add the source tree instead.
# The cadence toolchain uses -stdlib=libc++ (exceptions/RTTI/iostream available);
# the ISS is single-threaded, so disable pthreads.
if(NOT TARGET gtest)
  set(gtest_disable_pthreads
      ON
      CACHE BOOL "" FORCE
  )
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE
  )
  add_subdirectory(
    ${EXECUTORCH_ROOT}/third-party/googletest
    ${CMAKE_CURRENT_BINARY_DIR}/googletest
  )
endif()

# Seed the suite with one quantized op to prove the harness end-to-end on the
# ISS. Further op tests are added on top of this in the stack.
add_executable(
  cadence_hifi_op_tests
  test_op_quantized_relu_out.cpp
  ${EXECUTORCH_ROOT}/runtime/core/exec_aten/testing_util/tensor_util.cpp
)
target_compile_definitions(cadence_hifi_op_tests PRIVATE GTEST_HAS_PTHREAD=0)
target_include_directories(
  cadence_hifi_op_tests PRIVATE ${_common_include_directories}
                                ${CMAKE_BINARY_DIR}
)

# Direct-call: link the kernel lib that defines
# impl::HiFi::native::quantized_relu_out (custom_ops), the cadence kernels, and
# the core runtime (executorch provides the platform layer the tests'
# runtime_init() needs). No cadence_ops_lib registry is needed for direct-call
# tests.
target_link_libraries(
  cadence_hifi_op_tests PRIVATE gtest gtest_main gmock executorch custom_ops
                                cadence_kernels
)

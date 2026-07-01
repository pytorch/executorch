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

# Tame the vendored googletest build on the Xtensa clang toolchain. It hardcodes
# -Werror -Wconversion on its own sources, which clang-10 trips (e.g. an int ->
# signed char narrowing in gtest-port.h); -Wno-error keeps those non-fatal. A
# ccache launcher also preprocesses then recompiles the .i, where the -I dirs
# are unused and clang warns; -Qunused-arguments silences that. These land after
# googletest's own flags, so they win, and apply to this whole subtree.
add_compile_options(-Qunused-arguments -Wno-error)

# Let files say "include <executorch/path/to/header.h>".
set(_common_include_directories
    ${EXECUTORCH_ROOT}/.. ${EXECUTORCH_ROOT}/runtime/core/portable_type/c10
)

# Build googletest FOR the Xtensa target. find_package(GTest) would resolve the
# HOST gtest, which cannot link Xtensa objects, so add the source tree instead.
# The cadence toolchain uses -stdlib=libc++ (exceptions/RTTI/iostream
# available); the ISS is single-threaded, so disable pthreads.
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

# Op-level gtests, one source per operator. Each calls its kernel directly and
# checks with EXPECT_TENSOR_EQ, mirroring the BUCK tests.
add_executable(
  cadence_hifi_op_tests
  test_op_cat.cpp
  test_op_dequantize_per_tensor_out.cpp
  test_op_div.cpp
  test_op_im2row_out.cpp
  test_op_permute_copy.cpp
  test_op_quantize_per_tensor.cpp
  test_op_quantized_conv2d_out.cpp
  test_op_quantized_matmul_out.cpp
  test_op_quantized_relu_out.cpp
  test_op_transpose_copy.cpp
  ${EXECUTORCH_ROOT}/runtime/core/exec_aten/testing_util/tensor_util.cpp
)
target_compile_definitions(cadence_hifi_op_tests PRIVATE GTEST_HAS_PTHREAD=0)
target_include_directories(
  cadence_hifi_op_tests PRIVATE ${_common_include_directories}
                                ${CMAKE_BINARY_DIR}
)

# Direct-call: link the libs that define the kernels under test. custom_ops has
# the quantized ops; aten_ops_cadence has the aten-compliant ops (cat, div,
# im2row, permute, transpose); cadence_kernels is the shared nnlib layer;
# executorch provides the platform layer the tests' runtime_init() needs. No
# cadence_ops_lib registry is needed for direct-call tests.
target_link_libraries(
  cadence_hifi_op_tests
  PRIVATE gtest
          gtest_main
          gmock
          executorch
          custom_ops
          aten_ops_cadence
          cadence_kernels
)

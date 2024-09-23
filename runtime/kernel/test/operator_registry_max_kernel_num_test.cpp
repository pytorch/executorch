/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <vector>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::register_kernels;
using executorch::runtime::registry_has_op_function;

class OperatorRegistryMaxKernelNumTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// Register one kernel when max_kernel_num=1; success
TEST_F(OperatorRegistryMaxKernelNumTest, RegisterOneOp) {
  Kernel kernels[] = {Kernel("foo", [](KernelRuntimeContext&, EValue**) {})};
  auto s1 = register_kernels({kernels});
  EXPECT_EQ(s1, Error::Ok);
  EXPECT_FALSE(registry_has_op_function("fpp"));
  EXPECT_TRUE(registry_has_op_function("foo"));
}

// Register two kernels when max_kernel_num=1; fail
TEST_F(OperatorRegistryMaxKernelNumTest, RegisterTwoOpsFail) {
  Kernel kernels[] = {
      Kernel("foo1", [](KernelRuntimeContext&, EValue**) {}),
      Kernel("foo2", [](KernelRuntimeContext&, EValue**) {})};
  ET_EXPECT_DEATH(
      { (void)register_kernels({kernels}); },
      "The total number of kernels to be registered is larger than the limit 1");
}

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

namespace torch {
namespace executor {

class OperatorRegistryMaxKernelNumTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

// Register one kernel when max_kernel_num=1; success
TEST_F(OperatorRegistryMaxKernelNumTest, RegisterOneOp) {
  Kernel kernels[] = {Kernel("foo", [](RuntimeContext&, EValue**) {})};
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(kernels);
  auto s1 = register_kernels(kernels_array);
  EXPECT_EQ(s1, Error::Ok);
  EXPECT_FALSE(hasOpsFn("fpp"));
  EXPECT_TRUE(hasOpsFn("foo"));
}

// Register two kernels when max_kernel_num=1; fail
TEST_F(OperatorRegistryMaxKernelNumTest, RegisterTwoOpsFail) {
  Kernel kernels[] = {
      Kernel("foo1", [](RuntimeContext&, EValue**) {}),
      Kernel("foo2", [](RuntimeContext&, EValue**) {})};
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(kernels);
  ET_EXPECT_DEATH(
      { register_kernels(kernels_array); },
      "The total number of kernels to be registered is larger than the limit 1");
}

} // namespace executor
} // namespace torch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/test/RegisterKernels.h>

#include <gtest/gtest.h>

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;
using executorch::runtime::Error;
using executorch::runtime::registry_has_op_function;

class KernelManualRegistrationTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(KernelManualRegistrationTest, ManualRegister) {
  // Before registering, we can't find the add operator.
  EXPECT_FALSE(registry_has_op_function("aten::add.out"));

  // Call the generated registration function.
  Error result = torch::executor::register_all_kernels();
  EXPECT_EQ(result, Error::Ok);

  // We can now find the registered add operator.
  EXPECT_TRUE(registry_has_op_function("aten::add.out"));

  // We can't find a random other operator.
  EXPECT_FALSE(registry_has_op_function("fpp"));
}

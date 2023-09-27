/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Include RegisterKernels.h and call register_all_kernels().
#include <gtest/gtest.h>

#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/kernel/test/RegisterKernels.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;

namespace torch {
namespace executor {

class KernelManualRegistrationTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(KernelManualRegistrationTest, ManualRegister) {
  Error result = register_all_kernels();
  // Check that we can find the kernel for foo.
  EXPECT_EQ(result, Error::Ok);
  EXPECT_FALSE(hasOpsFn("fpp"));
  EXPECT_TRUE(hasOpsFn("aten::add.out"));
}

} // namespace executor
} // namespace torch

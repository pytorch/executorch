/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_runtime_context.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace ::testing;
using torch::executor::Error;
using torch::executor::KernelRuntimeContext;

class KernelRuntimeContextTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(KernelRuntimeContextTest, FailureStateDefaultsToOk) {
  KernelRuntimeContext context;

  EXPECT_EQ(context.failure_state(), Error::Ok);
}

TEST_F(KernelRuntimeContextTest, FailureStateReflectsFailure) {
  KernelRuntimeContext context;

  // Starts off Ok.
  EXPECT_EQ(context.failure_state(), Error::Ok);

  // Failing should update the failure state.
  context.fail(Error::MemoryAllocationFailed);
  EXPECT_EQ(context.failure_state(), Error::MemoryAllocationFailed);

  // State can be overwritten.
  context.fail(Error::Internal);
  EXPECT_EQ(context.failure_state(), Error::Internal);

  // And can be cleared.
  context.fail(Error::Ok);
  EXPECT_EQ(context.failure_state(), Error::Ok);
}

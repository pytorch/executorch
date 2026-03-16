/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>
#include <string>

using namespace ::testing;

using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Kernel;
using executorch::runtime::KernelRuntimeContext;
using executorch::runtime::register_kernels;
using executorch::runtime::Span;

class KernelDoubleRegistrationTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(KernelDoubleRegistrationTest, Basic) {
  Kernel kernels[] = {Kernel(
      "aten::add.out",
      "v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3",
      [](KernelRuntimeContext&, Span<EValue*>) {})};

  // First registration should succeed
  Error err = register_kernels({kernels});
  EXPECT_EQ(err, Error::Ok);

  // Second registration should succeed but skip the duplicate
  err = register_kernels({kernels});
  EXPECT_EQ(err, Error::Ok);
}

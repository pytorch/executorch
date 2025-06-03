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
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/test/utils/DeathTest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::runtime::Error;
using executorch::runtime::EValue;

class GeneratedLibAndAtenTest : public ::testing::Test {
 public:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(GeneratedLibAndAtenTest, GetKernelsFromATenRegistry) {
  // Check if the kernel exists in the ATen registry
  bool has_kernel =
      executorch::runtime::aten::registry_has_op_function("aten::add.out");
  EXPECT_TRUE(has_kernel)
      << "Kernel 'aten::add.out' not found in ATen registry";

  // Get the kernel from the ATen registry
  auto result =
      executorch::runtime::aten::get_op_function_from_registry("aten::add.out");
  EXPECT_EQ(result.error(), Error::Ok)
      << "Failed to get kernel from ATen registry";
  EXPECT_NE(*result, nullptr) << "Kernel function from ATen registry is null";
}

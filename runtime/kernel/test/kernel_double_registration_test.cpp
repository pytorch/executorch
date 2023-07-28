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

namespace torch {
namespace executor {

class KernelDoubleRegistrationTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(KernelDoubleRegistrationTest, Basic) {
  Kernel kernels[] = {Kernel(
      "aten::add.out",
      "v1/7;0,1,2,3|7;0,1,2,3|7;0,1,2,3",
      [](RuntimeContext&, EValue**) {})};
  ArrayRef<Kernel> kernels_array = ArrayRef<Kernel>(kernels);
  Error err = Error::InvalidArgument;

  ET_EXPECT_DEATH(
      { auto res = register_kernels(kernels_array); },
      std::to_string(static_cast<uint32_t>(err)));
}

} // namespace executor
} // namespace torch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>
#include <executorch/runtime/kernel/operator_registry.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace torch {
namespace executor {

using namespace ::testing;

Tensor& my_op_out(const Tensor& a, Tensor& out) {
  (void)a;
  return out;
}

Tensor& set_1_out(Tensor& out) {
  out.mutable_data_ptr<int32_t>()[0] = 1;
  return out;
}

class MakeATenFunctorFromETFunctorTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(MakeATenFunctorFromETFunctorTest, Basic) {
  auto function = wrapper_impl<decltype(&my_op_out), my_op_out>::wrap;
  at::Tensor a = torch::tensor({1.0f});
  at::Tensor b = torch::tensor({2.0f});
  at::Tensor c = function(a, b);
  EXPECT_EQ(c.const_data_ptr<float>()[0], 2.0f);
}

} // namespace executor
} // namespace torch

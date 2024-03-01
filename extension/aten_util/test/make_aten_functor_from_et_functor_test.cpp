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

Tensor& add_1_out(const Tensor& a, Tensor& out) {
  (void)a;
  out.mutable_data_ptr<int32_t>()[0] += 1;
  return out;
}

class MakeATenFunctorFromETFunctorTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(MakeATenFunctorFromETFunctorTest, Basic) {
  auto function = WRAP(my_op_out, 1);
  at::Tensor a = torch::tensor({1.0f});
  at::Tensor b = torch::tensor({2.0f});
  at::Tensor c = function(a, b);
  EXPECT_EQ(c.const_data_ptr<float>()[0], 2.0f);
}

TORCH_LIBRARY(my_op, m) {
  m.def("add_1.out", WRAP(add_1_out, 1));
};

TEST_F(MakeATenFunctorFromETFunctorTest, RegisterWrappedFunction) {
  auto op = c10::Dispatcher::singleton().findSchema({"my_op::add_1", "out"});
  EXPECT_TRUE(op.has_value());
  at::Tensor a =
      torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor b =
      torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
  torch::jit::Stack stack = {a, b};
  op.value().callBoxed(&stack);
  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int32_t>()[0], 3);
}

} // namespace executor
} // namespace torch

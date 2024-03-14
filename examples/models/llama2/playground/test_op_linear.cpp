/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <torch/library.h>

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

Tensor& quantized_embedding_byte_out(
    const Tensor& weight,
    const Tensor& weight_scales,
    const Tensor& weight_zero_points,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    const Tensor& indices,
    Tensor& out) {
  (void)weight;
  (void)weight_scales;
  (void)weight_zero_points;
  (void)weight_quant_min;
  (void)indices;
  out.mutable_data_ptr<int32_t>()[0] -= static_cast<int32_t>(weight_quant_max);
  return out;
}

class MakeATenFunctorFromETFunctorTest : public ::testing::Test {
 public:
  void SetUp() override {
    torch::executor::runtime_init();
  }
};

TEST_F(MakeATenFunctorFromETFunctorTest, Basic) {
  auto function = WRAP_TO_ATEN(my_op_out, 1);
  at::Tensor a = torch::tensor({1.0f});
  at::Tensor b = torch::tensor({2.0f});
  at::Tensor c = function(a, b);
  EXPECT_EQ(c.const_data_ptr<float>()[0], 2.0f);
}

TORCH_LIBRARY(my_op, m) {
  m.def("add_1.out", WRAP_TO_ATEN(add_1_out, 1));
  m.def(
      "embedding_byte.out(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(quantized_embedding_byte_out, 6));
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

TEST_F(MakeATenFunctorFromETFunctorTest, TestEmbeddingByte) {
  auto op =
      c10::Dispatcher::singleton().findSchema({"my_op::embedding_byte", "out"});
  EXPECT_TRUE(op.has_value());
  at::Tensor weight =
      torch::tensor({1}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor scale =
      torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor zero_point =
      torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor indices =
      torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
  at::Tensor out =
      torch::tensor({4}, torch::TensorOptions().dtype(torch::kInt32));
  torch::jit::Stack stack = {weight, scale, zero_point, 0, 1, indices, out};
  op.value().callBoxed(&stack);
  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int32_t>()[0], 3);
}

} // namespace executor
} // namespace torch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <torch/library.h>

using namespace ::testing;
using ::executorch::extension::internal::type_convert;
using ::executorch::extension::internal::type_map;
using ::torch::executor::ScalarType;
using ::torch::executor::Tensor;

Tensor& my_op_out(const Tensor& a, Tensor& out) {
  (void)a;
  return out;
}

Tensor& add_1_out(const Tensor& a, Tensor& out) {
  (void)a;
  out.mutable_data_ptr<int32_t>()[0] += 1;
  return out;
}

Tensor& add_optional_scalar_out(
    torch::executor::optional<int64_t> s1,
    torch::executor::optional<int64_t> s2,
    Tensor& out) {
  if (s1.has_value()) {
    out.mutable_data_ptr<int64_t>()[0] += s1.value();
  }
  if (s2.has_value()) {
    out.mutable_data_ptr<int64_t>()[0] += s2.value();
  }
  return out;
}

Tensor& add_optional_tensor_out(
    torch::executor::optional<torch::executor::Tensor> s1,
    torch::executor::optional<torch::executor::Tensor> s2,
    Tensor& out) {
  if (s1.has_value()) {
    out.mutable_data_ptr<int64_t>()[0] +=
        s1.value().mutable_data_ptr<int64_t>()[0];
  }
  if (s2.has_value()) {
    out.mutable_data_ptr<int64_t>()[0] +=
        s2.value().mutable_data_ptr<int64_t>()[0];
  }
  return out;
}

Tensor& sum_arrayref_scalar_out(
    torch::executor::ArrayRef<int64_t> a,
    Tensor& out) {
  for (int i = 0; i < a.size(); i++) {
    out.mutable_data_ptr<int64_t>()[0] += a[i];
  }
  return out;
}

Tensor& sum_arrayref_tensor_out(
    torch::executor::ArrayRef<torch::executor::Tensor> a,
    Tensor& out) {
  for (int i = 0; i < a.size(); i++) {
    out.mutable_data_ptr<int32_t>()[0] += a[i].const_data_ptr<int32_t>()[0];
  }
  return out;
}

Tensor& sum_arrayref_optional_tensor_out(
    torch::executor::ArrayRef<
        torch::executor::optional<torch::executor::Tensor>> a,
    Tensor& out) {
  for (int i = 0; i < a.size(); i++) {
    if (a[i].has_value()) {
      out.mutable_data_ptr<int32_t>()[0] +=
          a[i].value().const_data_ptr<int32_t>()[0];
    }
  }
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

TEST_F(MakeATenFunctorFromETFunctorTest, TestTypeMap_Scalar) {
  EXPECT_TRUE((std::is_same<type_map<int64_t>::type, int64_t>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestTypeMap_Tensor) {
  // Normal, ref, const, and const ref.
  EXPECT_TRUE(
      (std::is_same<type_map<torch::executor::Tensor>::type, at::Tensor>::
           value));
  EXPECT_TRUE(
      (std::is_same<type_map<torch::executor::Tensor&>::type, at::Tensor&>::
           value));
  EXPECT_TRUE((std::is_same<
               type_map<const torch::executor::Tensor>::type,
               const at::Tensor>::value));
  EXPECT_TRUE((std::is_same<
               type_map<const torch::executor::Tensor&>::type,
               const at::Tensor&>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestTypeMap_Optionals) {
  // Scalar.
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::optional<int64_t>>::type,
               c10::optional<int64_t>>::value));
  // Tensor.
  EXPECT_TRUE(
      (std::is_same<
          type_map<torch::executor::optional<torch::executor::Tensor>>::type,
          c10::optional<at::Tensor>>::value));
  // ArrayRef.
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::optional<
                   torch::executor::ArrayRef<int64_t>>>::type,
               c10::optional<c10::ArrayRef<int64_t>>>::value));
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::optional<
                   torch::executor::ArrayRef<torch::executor::Tensor>>>::type,
               c10::optional<c10::ArrayRef<at::Tensor>>>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestTypeMap_ArrayRef) {
  // Scalar.
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::ArrayRef<int64_t>>::type,
               c10::ArrayRef<int64_t>>::value));
  // Tensor.
  EXPECT_TRUE(
      (std::is_same<
          type_map<torch::executor::ArrayRef<torch::executor::Tensor>>::type,
          c10::ArrayRef<at::Tensor>>::value));
  // Optionals.
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::ArrayRef<
                   torch::executor::optional<int64_t>>>::type,
               c10::ArrayRef<c10::optional<int64_t>>>::value));
  EXPECT_TRUE((std::is_same<
               type_map<torch::executor::ArrayRef<
                   torch::executor::optional<torch::executor::Tensor>>>::type,
               c10::ArrayRef<c10::optional<at::Tensor>>>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestConvert_Tensor) {
  // Convert at to et.
  at::Tensor at_in = torch::tensor({1});
  auto et = type_convert<at::Tensor, torch::executor::Tensor>(at_in).call();
  EXPECT_TRUE((std::is_same<decltype(et), torch::executor::Tensor>::value));

  // Convert et to at.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  torch::executor::Tensor et_in = tf.ones({3});
  auto at_out = type_convert<torch::executor::Tensor, at::Tensor>(et_in).call();
  EXPECT_TRUE((std::is_same<decltype(at_out), at::Tensor>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestConvert_OptionalScalar) {
  // Convert optional at to et.
  auto optional_at_in = c10::optional<int64_t>();
  auto optional_et =
      type_convert<c10::optional<int64_t>, torch::executor::optional<int64_t>>(
          optional_at_in)
          .call();
  EXPECT_TRUE(
      (std::is_same<decltype(optional_et), torch::executor::optional<int64_t>>::
           value));

  // Convert optional et to at.
  auto optional_et_in = torch::executor::optional<int64_t>();
  auto optional_at_out =
      type_convert<torch::executor::optional<int64_t>, c10::optional<int64_t>>(
          optional_et_in)
          .call();
  EXPECT_TRUE(
      (std::is_same<decltype(optional_at_out), c10::optional<int64_t>>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestConvert_OptionalTensor) {
  // Convert optional at to et.
  auto optional_at_in = c10::optional<at::Tensor>();
  auto optional_et =
      type_convert<
          c10::optional<at::Tensor>,
          torch::executor::optional<torch::executor::Tensor>>(optional_at_in)
          .call();
  EXPECT_TRUE((std::is_same<
               decltype(optional_et),
               torch::executor::optional<torch::executor::Tensor>>::value));

  // Convert optional et to at.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  auto et_in = torch::executor::optional<torch::executor::Tensor>(tf.ones({3}));
  auto optional_at_out = type_convert<
                             torch::executor::optional<torch::executor::Tensor>,
                             c10::optional<at::Tensor>>(optional_et)
                             .call();
  EXPECT_TRUE(
      (std::is_same<decltype(optional_at_out), c10::optional<at::Tensor>>::
           value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestConvert_ArrayRefScalar) {
  // Convert arrayref at to et.
  const std::vector<int64_t> vec = {1, 2, 3};
  c10::ArrayRef<int64_t> arrayref_at_in = c10::ArrayRef<int64_t>(vec);
  auto arrayref_et =
      type_convert<c10::ArrayRef<int64_t>, torch::executor::ArrayRef<int64_t>>(
          arrayref_at_in)
          .call();
  EXPECT_TRUE(
      (std::is_same<decltype(arrayref_et), torch::executor::ArrayRef<int64_t>>::
           value));

  // Convert array ref et to at.
  auto arrayref_et_in =
      torch::executor::ArrayRef<int64_t>(vec.data(), vec.size());

  auto arrayref_at_out =
      type_convert<torch::executor::ArrayRef<int64_t>, c10::ArrayRef<int64_t>>(
          arrayref_et_in)
          .call();
  EXPECT_TRUE(
      (std::is_same<decltype(arrayref_at_out), c10::ArrayRef<int64_t>>::value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestConvert_ArrayRefTensor) {
  // Convert arrayref at to et.
  const std::vector<at::Tensor> vec_at{torch::tensor({1}), torch::tensor({2})};
  c10::ArrayRef<at::Tensor> arrayref_at_in =
      c10::ArrayRef<at::Tensor>(vec_at.data(), vec_at.size());

  auto arrayref_et =
      type_convert<
          c10::ArrayRef<at::Tensor>,
          torch::executor::ArrayRef<torch::executor::Tensor>>(arrayref_at_in)
          .call();
  EXPECT_TRUE((std::is_same<
               decltype(arrayref_et),
               torch::executor::ArrayRef<torch::executor::Tensor>>::value));
  // Convert array ref et to at.
  torch::executor::testing::TensorFactory<ScalarType::Int> tf;
  std::vector<torch::executor::Tensor> vec_et{tf.ones({1}), tf.ones({2})};
  auto arrayref_et_in = torch::executor::ArrayRef<torch::executor::Tensor>(
      vec_et.data(), vec_et.size());

  auto arrayref_at_out = type_convert<
                             torch::executor::ArrayRef<torch::executor::Tensor>,
                             c10::ArrayRef<at::Tensor>>(arrayref_et_in)
                             .call();
  EXPECT_TRUE(
      (std::is_same<decltype(arrayref_at_out), c10::ArrayRef<at::Tensor>>::
           value));
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_Basic) {
  auto function = WRAP_TO_ATEN(my_op_out, 1);
  at::Tensor a = torch::tensor({1.0f});
  at::Tensor b = torch::tensor({2.0f});
  at::Tensor c = function(a, b);
  EXPECT_EQ(c.const_data_ptr<float>()[0], 2.0f);
}

// Register operators.
TORCH_LIBRARY(my_op, m) {
  m.def("add_1.out", WRAP_TO_ATEN(add_1_out, 1));
  m.def(
      "embedding_byte.out(Tensor weight, Tensor weight_scales, Tensor weight_zero_points, int weight_quant_min, int weight_quant_max, Tensor indices, *, Tensor(a!) out) -> Tensor(a!)",
      WRAP_TO_ATEN(quantized_embedding_byte_out, 6));
  m.def("add_optional_scalar.out", WRAP_TO_ATEN(add_optional_scalar_out, 2));
  m.def("add_optional_tensor.out", WRAP_TO_ATEN(add_optional_tensor_out, 2));
  m.def("sum_arrayref_scalar.out", WRAP_TO_ATEN(sum_arrayref_scalar_out, 1));
  m.def("sum_arrayref_tensor.out", WRAP_TO_ATEN(sum_arrayref_tensor_out, 1));
  m.def(
      "sum_arrayref_optional_tensor.out",
      WRAP_TO_ATEN(sum_arrayref_optional_tensor_out, 1));
};

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_RegisterWrappedFunction) {
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

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_EmbeddingByte) {
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

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_OptionalScalarAdd) {
  c10::optional<int64_t> a = c10::optional<int64_t>(3);
  c10::optional<int64_t> b = c10::optional<int64_t>();
  at::Tensor out = torch::tensor({0});

  auto op = c10::Dispatcher::singleton().findSchema(
      {"my_op::add_optional_scalar", "out"});
  EXPECT_TRUE(op.has_value());
  torch::jit::Stack stack = {a, b, out};
  op.value().callBoxed(&stack);

  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int64_t>()[0], 3);
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_OptionalTensorAdd) {
  c10::optional<at::Tensor> a = c10::optional<at::Tensor>(torch::tensor({8}));
  c10::optional<at::Tensor> b = c10::optional<at::Tensor>();
  at::Tensor out = torch::tensor({0});

  auto op = c10::Dispatcher::singleton().findSchema(
      {"my_op::add_optional_tensor", "out"});
  EXPECT_TRUE(op.has_value());
  torch::jit::Stack stack = {a, b, out};
  op.value().callBoxed(&stack);

  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int64_t>()[0], 8);
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_ArrayRefScalarAdd) {
  std::vector<int64_t> vec{2, 3, 4};
  at::ArrayRef<int64_t> arrayref = at::ArrayRef(vec.data(), vec.size());
  at::Tensor out = torch::tensor({0});

  auto op = c10::Dispatcher::singleton().findSchema(
      {"my_op::sum_arrayref_scalar", "out"});
  EXPECT_TRUE(op.has_value());
  torch::jit::Stack stack = {arrayref, out};
  op.value().callBoxed(&stack);

  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int64_t>()[0], 9);
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_ArrayRefTensorAdd) {
  std::vector<at::Tensor> vec{
      torch::tensor({1}), torch::tensor({2}), torch::tensor({3})};
  at::ArrayRef arrayref = at::ArrayRef(vec.data(), vec.size());
  at::Tensor out = torch::tensor({0});

  auto op = c10::Dispatcher::singleton().findSchema(
      {"my_op::sum_arrayref_tensor", "out"});
  EXPECT_TRUE(op.has_value());
  torch::jit::Stack stack = {arrayref, out};
  op.value().callBoxed(&stack);

  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int64_t>()[0], 6);
}

TEST_F(MakeATenFunctorFromETFunctorTest, TestWrap_ArrayRefOptional) {
  std::vector<c10::optional<at::Tensor>> vec{
      c10::optional<at::Tensor>(torch::tensor({1})),
      c10::optional<at::Tensor>(),
      c10::optional<at::Tensor>(torch::tensor({3}))};
  at::Tensor out = torch::tensor({0});

  at::ArrayRef arrayref = at::ArrayRef(vec.data(), vec.size());
  auto op = c10::Dispatcher::singleton().findSchema(
      {"my_op::sum_arrayref_optional_tensor", "out"});
  EXPECT_TRUE(op.has_value());
  torch::jit::Stack stack = {arrayref, out};
  op.value().callBoxed(&stack);

  EXPECT_EQ(stack.size(), 1);
  EXPECT_EQ(stack[0].toTensor().const_data_ptr<int64_t>()[0], 4);
}

// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/MethodMeta.h>

#include <stdexcept>

#include <gtest/gtest.h>

namespace ptn {
namespace {

Method make_tensor_method() {
  Method method;
  method.name = "forward";
  method.graph.values.emplace_back("input", TensorMeta{kFloat, {2, 3}, {1, 0}});
  method.graph.values.emplace_back("mutation", TensorMeta{kFloat, {2, 3}, {}});
  method.graph.values.emplace_back("output", TensorMeta{kFloat, {2, 3}, {}});
  method.graph.input_ids = {0};
  method.graph.output_ids = {1, 2};
  method.output_specs = {
      OutputSpec{OutputKind::BufferMutation, /*target_id=*/0},
      OutputSpec{OutputKind::UserOutput, kInvalid},
  };
  return method;
}

TEST(MethodMetaTest, FromMethod_TensorSignature_PreservesMetadata) {
  const MethodMeta meta = MethodMeta::from_method(make_tensor_method());

  ASSERT_EQ(meta.inputs().size(), 1);
  ASSERT_EQ(meta.outputs().size(), 1);
  EXPECT_EQ(meta.name(), "forward");
  EXPECT_EQ(
      std::vector<int64_t>(
          meta.inputs()[0].sizes().begin(), meta.inputs()[0].sizes().end()),
      (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(
      std::vector<uint8_t>(
          meta.inputs()[0].dim_order().begin(),
          meta.inputs()[0].dim_order().end()),
      (std::vector<uint8_t>{1, 0}));
  EXPECT_EQ(
      std::vector<int64_t>(
          meta.inputs()[0].strides().begin(), meta.inputs()[0].strides().end()),
      (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(meta.inputs()[0].dtype(), kFloat);
  EXPECT_EQ(meta.inputs()[0].numel(), 6);
  EXPECT_EQ(meta.inputs()[0].nbytes(), 24);
  EXPECT_EQ(
      std::vector<int64_t>(
          meta.outputs()[0].strides().begin(),
          meta.outputs()[0].strides().end()),
      (std::vector<int64_t>{3, 1}));
}

TEST(MethodMetaTest, FromMethod_NonTensorInput_Throws) {
  Method method = make_tensor_method();
  method.graph.input_ids = {3};
  method.graph.values.emplace_back("scalar", Scalar{1});

  EXPECT_THROW(MethodMeta::from_method(method), std::runtime_error);
}

TEST(MethodMetaTest, FromMethod_InvalidInputId_Throws) {
  Method method = make_tensor_method();
  method.graph.input_ids = {/*value=*/9};

  EXPECT_THROW(MethodMeta::from_method(method), std::runtime_error);
}

TEST(MethodMetaTest, FromMethod_MismatchedOutputSpecs_Throws) {
  Method method = make_tensor_method();
  method.output_specs.pop_back();

  EXPECT_THROW(MethodMeta::from_method(method), std::runtime_error);
}

TEST(MethodMetaTest, FromMethod_NonPermutationDimOrder_Throws) {
  Method method = make_tensor_method();
  method.graph.values[0] = Value("input", TensorMeta{kFloat, {2, 3}, {0, 0}});

  EXPECT_THROW(MethodMeta::from_method(method), std::runtime_error);
}

TEST(MethodMetaTest, FromMethod_OwnsMetadata) {
  Method method = make_tensor_method();
  const MethodMeta meta = MethodMeta::from_method(method);

  method.name = "changed";
  method.graph.values[0] = Value("changed", TensorMeta{kInt, {1}, {}});

  EXPECT_EQ(meta.name(), "forward");
  EXPECT_EQ(
      std::vector<int64_t>(
          meta.inputs()[0].sizes().begin(), meta.inputs()[0].sizes().end()),
      (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(meta.inputs()[0].dtype(), kFloat);
}

} // namespace
} // namespace ptn

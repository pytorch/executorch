// cppcheck-suppress-file syntaxError

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/extension/module/MethodMetaBridge.h>

#include <array>
#include <limits>
#include <utility>
#include <vector>

#include <executorch/runtime/core/tag.h>
#include <gtest/gtest.h>

namespace executorch::extension::native_module {
namespace {

ptn::Method make_method(ptn::ScalarType dtype = ptn::kFloat) {
  ptn::Method method;
  method.name = "forward";
  method.graph.values.emplace_back(
      "input", ptn::TensorMeta{dtype, {2, 3}, {1, 0}});
  method.graph.values.emplace_back(
      "output", ptn::TensorMeta{ptn::kFloat, {2, 3}, {}});
  method.graph.input_ids = {0};
  method.graph.output_ids = {1};
  method.output_specs = {
      ptn::OutputSpec{ptn::OutputKind::UserOutput, ptn::kInvalid}};
  return method;
}

TEST(MethodMetaBridgeTest, MapsEveryScalarType) {
  using EtScalarType = executorch::aten::ScalarType;
  constexpr std::array cases{
      std::pair{ptn::kByte, EtScalarType::Byte},
      std::pair{ptn::kChar, EtScalarType::Char},
      std::pair{ptn::kShort, EtScalarType::Short},
      std::pair{ptn::kInt, EtScalarType::Int},
      std::pair{ptn::kLong, EtScalarType::Long},
      std::pair{ptn::kHalf, EtScalarType::Half},
      std::pair{ptn::kFloat, EtScalarType::Float},
      std::pair{ptn::kDouble, EtScalarType::Double},
      std::pair{ptn::kBool, EtScalarType::Bool},
      std::pair{ptn::kBFloat16, EtScalarType::BFloat16},
      std::pair{ptn::kUInt16, EtScalarType::UInt16},
      std::pair{ptn::kUInt32, EtScalarType::UInt32},
      std::pair{ptn::kUInt64, EtScalarType::UInt64},
  };

  for (const auto& [native_type, et_type] : cases) {
    const auto bridge = MethodMetaBridge::create(
        ptn::MethodMeta::from_method(make_method(native_type)));
    const auto input = bridge->view().input_tensor_meta(/*index=*/0);
    ASSERT_TRUE(input.ok());
    EXPECT_EQ(input->scalar_type(), et_type);
  }
}

TEST(MethodMetaBridgeTest, RejectsUnrepresentableMetadata) {
  ptn::Method invalid_type = make_method(static_cast<ptn::ScalarType>(127));
  EXPECT_THROW(
      MethodMetaBridge::create(ptn::MethodMeta::from_method(invalid_type)),
      std::runtime_error);

  ptn::Method oversized = make_method();
  oversized.graph.values[0] = ptn::Value(
      "input",
      ptn::TensorMeta{
          ptn::kByte,
          {static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1},
          {}});
  EXPECT_THROW(
      MethodMetaBridge::create(ptn::MethodMeta::from_method(oversized)),
      std::runtime_error);
}

TEST(MethodMetaBridgeTest, PreservesTensorSignatureWithoutFakeRuntimeData) {
  const ptn::Method method = make_method();
  const ptn::MethodMeta native_meta = ptn::MethodMeta::from_method(method);
  const auto storage = MethodMetaBridge::create(native_meta);
  const auto meta = storage->view();

  EXPECT_STREQ(meta.name(), "forward");
  ASSERT_EQ(meta.num_inputs(), 1);
  ASSERT_EQ(meta.num_outputs(), 1);
  ASSERT_TRUE(meta.input_tag(0).ok());
  EXPECT_EQ(*meta.input_tag(0), runtime::Tag::Tensor);
  ASSERT_TRUE(meta.output_tag(0).ok());
  EXPECT_EQ(*meta.output_tag(0), runtime::Tag::Tensor);

  const auto input = meta.input_tensor_meta(0);
  ASSERT_TRUE(input.ok());
  EXPECT_TRUE(input->name().empty());
  const std::vector<executorch::aten::SizesType> sizes(
      input->sizes().begin(), input->sizes().end());
  EXPECT_EQ(sizes, (std::vector<executorch::aten::SizesType>{2, 3}));
  const std::vector<executorch::aten::DimOrderType> dim_order(
      input->dim_order().begin(), input->dim_order().end());
  EXPECT_EQ(dim_order, (std::vector<executorch::aten::DimOrderType>{1, 0}));
  EXPECT_EQ(input->scalar_type(), executorch::aten::ScalarType::Float);
  EXPECT_EQ(input->nbytes(), 24);
  EXPECT_FALSE(input->is_memory_planned());

  const auto output = meta.output_tensor_meta(0);
  ASSERT_TRUE(output.ok());
  EXPECT_TRUE(output->name().empty());
  EXPECT_EQ(meta.num_attributes(), 0);
  EXPECT_EQ(meta.num_memory_planned_buffers(), 0);
  EXPECT_EQ(meta.num_backends(), 0);
  EXPECT_FALSE(meta.uses_backend("NativeBackend"));
  EXPECT_EQ(meta.num_instructions(), 0);
}

TEST(MethodMetaBridgeTest, MapsUnsignedTypesAcrossDifferentEnumValues) {
  ptn::Method method = make_method();
  method.graph.values[0] =
      ptn::Value("input", ptn::TensorMeta{ptn::kUInt16, {2}, {}});
  const auto storage =
      MethodMetaBridge::create(ptn::MethodMeta::from_method(method));

  const auto input = storage->view().input_tensor_meta(0);
  ASSERT_TRUE(input.ok());
  EXPECT_EQ(input->scalar_type(), executorch::aten::ScalarType::UInt16);
}

} // namespace
} // namespace executorch::extension::native_module

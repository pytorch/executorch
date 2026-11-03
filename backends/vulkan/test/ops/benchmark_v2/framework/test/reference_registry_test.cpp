// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>

#include <gtest/gtest.h>

#include <executorch/runtime/core/portable_type/half.h>
#include <cstring>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

reference::Tensor
identity_reference(const Json&, const reference::TensorMap& tensors, size_t) {
  return tensors.at("x");
}

reference::Tensor offset_reference(
    const Json&,
    const reference::TensorMap& tensors,
    size_t threads) {
  reference::Tensor result = tensors.at("x");
  for (double& value : result.values) {
    value += threads;
  }
  return result;
}

ReferenceOutputs
multiple_reference(const Json&, const reference::TensorMap&, size_t) {
  return {{"scale", {0.12345}}, {"zero_point", {-3}}};
}

REGISTER_REFERENCE_FUNCTIONS {
  REGISTER_REFERENCE_FN(test_identity, identity_reference);
  REGISTER_REFERENCE_OUTPUTS_FN(test_multiple, multiple_reference);
  REGISTER_REFERENCE_FN(test_offset, offset_reference);
  REGISTER_OPERATOR_REFERENCE(aten.test_reference.default, test_identity);
  REGISTER_SHADER_REFERENCE(test_identity_shader, test_identity);
}

TEST(BenchmarkReferenceRegistry, RegisteredNamesSelectCallables) {
  ReferenceRegistry registry;
  registry.register_reference("identity", identity_reference);
  registry.register_reference("offset", offset_reference);
  EXPECT_EQ(registry.get_reference("identity"), identity_reference);
  EXPECT_EQ(registry.get_reference("offset"), offset_reference);
}

TEST(BenchmarkReferenceRegistry, MultipleNamesCanShareAReference) {
  ReferenceRegistry registry;
  registry.register_reference("identity", identity_reference);
  registry.register_operator_reference("aten.first.default", "identity");
  registry.register_operator_reference("aten.second.default", "identity");
  registry.register_shader_reference("shader_a", "identity");
  registry.register_shader_reference("shader_b", "identity");
  EXPECT_EQ(registry.operator_reference("aten.first.default"), "identity");
  EXPECT_EQ(registry.operator_reference("aten.second.default"), "identity");
  EXPECT_EQ(registry.shader_reference("shader_a"), "identity");
  EXPECT_EQ(registry.shader_reference("shader_b"), "identity");
}

TEST(BenchmarkReferenceRegistry, AssociationsCanPrecedeFunctionRegistration) {
  ReferenceRegistry registry;
  registry.register_operator_reference("aten.later.default", "later");
  registry.register_shader_reference("later_shader", "later");
  EXPECT_THROW(
      registry.operator_reference("aten.later.default"), std::invalid_argument);
  EXPECT_THROW(
      registry.shader_reference("later_shader"), std::invalid_argument);
  registry.register_reference("later", identity_reference);
  EXPECT_EQ(registry.operator_reference("aten.later.default"), "later");
  EXPECT_EQ(registry.shader_reference("later_shader"), "later");
}

TEST(BenchmarkReferenceRegistry, OperatorAndShaderNamespacesAreSeparate) {
  ReferenceRegistry registry;
  registry.register_reference("identity", identity_reference);
  registry.register_reference("offset", offset_reference);
  registry.register_operator_reference("same_name", "identity");
  registry.register_shader_reference("same_name", "offset");
  EXPECT_EQ(registry.operator_reference("same_name"), "identity");
  EXPECT_EQ(registry.shader_reference("same_name"), "offset");
}

TEST(BenchmarkReferenceRegistry, DuplicateFunctionNamesKeepOriginalFunction) {
  ReferenceRegistry registry;
  registry.register_reference("identity", identity_reference);
  EXPECT_THROW(
      registry.register_reference("identity", offset_reference),
      std::invalid_argument);
  EXPECT_EQ(registry.get_reference("identity"), identity_reference);
}

TEST(BenchmarkReferenceRegistry, DuplicateAssociationsKeepOriginalMapping) {
  ReferenceRegistry registry;
  registry.register_reference("identity", identity_reference);
  registry.register_reference("offset", offset_reference);
  registry.register_operator_reference("aten.same.default", "identity");
  registry.register_shader_reference("same_shader", "identity");
  EXPECT_THROW(
      registry.register_operator_reference("aten.same.default", "offset"),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_reference("same_shader", "offset"),
      std::invalid_argument);
  EXPECT_EQ(registry.operator_reference("aten.same.default"), "identity");
  EXPECT_EQ(registry.shader_reference("same_shader"), "identity");
}

TEST(BenchmarkReferenceRegistry, MissingFunctionsAndMappingsFailLookup) {
  ReferenceRegistry registry;
  EXPECT_THROW(registry.get_reference("unknown"), std::invalid_argument);
  EXPECT_THROW(registry.operator_reference("unknown"), std::invalid_argument);
  EXPECT_THROW(registry.shader_reference("unknown"), std::invalid_argument);
}

TEST(BenchmarkReferenceRegistry, RejectsEmptyNamesAndNullFunctions) {
  ReferenceRegistry registry;
  EXPECT_THROW(
      registry.register_reference("", identity_reference),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_reference("null", nullptr), std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_reference("", "identity"),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_reference("op", ""), std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_reference("", "identity"),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_reference("shader", ""), std::invalid_argument);
}

Json registered_case(bool shader) {
  Json spec = {
      {"tensors", {{"x", {{"sizes", {2}}}}, {"out", {{"sizes", {2}}}}}}};
  if (shader) {
    spec["shader"] = "test_identity_shader";
    spec["bindings"] = {{{"tensor", "out"}}, {{"tensor", "x"}}};
  } else {
    spec["operator"] = "aten.test_reference.default";
    spec["args"] = {{{"tensor", "x"}}, {{"tensor", "out"}}};
  }
  return spec;
}

TEST(
    BenchmarkReferenceRegistry,
    StaticAssociationsDispatchRegisteredFunctions) {
  const std::vector<float> values = {1.5f, -2.0f};
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  for (bool shader : {false, true}) {
    const Json spec = registered_case(shader);
    EXPECT_EQ(reference_name(spec), "test_identity");
    EXPECT_EQ(
        reference_outputs(spec, {}, {{"x", bytes}}).at("out"),
        (std::vector<double>{1.5, -2.0}));
  }
}

TEST(
    BenchmarkReferenceRegistry,
    JsonOverrideSelectsCallableAndForwardsThreads) {
  Json spec = registered_case(false);
  spec["reference"] = "test_offset";
  const std::vector<float> values = {1, 2};
  std::vector<uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  EXPECT_EQ(reference_name(spec), "test_offset");
  EXPECT_EQ(
      reference_outputs(spec, {}, {{"x", bytes}}, 3).at("out"),
      (std::vector<double>{4, 5}));
  spec["reference"] = "not_registered";
  EXPECT_THROW(reference_name(spec), std::invalid_argument);
}

TEST(BenchmarkReferenceRegistry, BuiltinRegistrationsAreLinkedFromFamilyFiles) {
  const auto& registry = reference_registry();
  EXPECT_EQ(registry.operator_reference("aten.linear.default"), "linear");
  EXPECT_EQ(registry.operator_reference("aten.convolution.default"), "conv2d");
  EXPECT_EQ(
      registry.operator_reference("et_vk.q8ta_conv2d.default"), "q8ta_conv2d");
  EXPECT_EQ(
      registry.operator_reference("et_vk.q4gsw_linear.default"),
      "q4gsw_linear");
  EXPECT_EQ(registry.shader_reference("buffer_to_buffer_float"), "copy");
  EXPECT_EQ(registry.shader_reference("buffer_to_buffer_half"), "copy");
  EXPECT_EQ(registry.shader_reference("binary_add_texture3d_float"), "add");
  EXPECT_EQ(registry.shader_reference("binary_add_texture3d_half"), "add");
}

} // namespace
TEST(BenchmarkReferenceRegistry, MultiOutputReferencesShareTheNameNamespace) {
  ReferenceRegistry registry;
  registry.register_reference_outputs("multiple", multiple_reference);
  registry.register_operator_reference("multi_op", "multiple");
  EXPECT_TRUE(registry.has_reference("multiple"));
  EXPECT_EQ(registry.operator_reference("multi_op"), "multiple");
  EXPECT_EQ(registry.get_reference_outputs("multiple"), multiple_reference);
  EXPECT_THROW(
      registry.register_reference("multiple", identity_reference),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_reference_outputs("multiple", multiple_reference),
      std::invalid_argument);
  registry.register_reference("single", identity_reference);
  EXPECT_EQ(registry.get_reference_outputs("single"), nullptr);
  EXPECT_THROW(
      registry.register_reference_outputs("single", multiple_reference),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_reference_outputs("", multiple_reference),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_reference_outputs("null", nullptr),
      std::invalid_argument);
}

TEST(
    BenchmarkReferenceRegistry,
    MultiOutputReferencesValidateCountsAndRoundPerDtype) {
  Json spec{
      {"operator", "test_multi"},
      {"reference", "test_multiple"},
      {"tensors",
       {{"scale", {{"sizes", {1}}, {"dtype", "float16"}}},
        {"zero_point", {{"sizes", {1}}, {"dtype", "int8"}}}}}};
  const auto results = reference_outputs(spec, Json::object(), {});
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(
      results.at("scale")[0],
      float(executorch::runtime::etensor::Half(0.12345f)));
  EXPECT_EQ(results.at("zero_point")[0], -3);
  spec["tensors"]["scale"]["sizes"] = {2};
  EXPECT_THROW(
      reference_outputs(spec, Json::object(), {}), std::invalid_argument);
  spec["tensors"].erase("scale");
  EXPECT_THROW(
      reference_outputs(spec, Json::object(), {}), std::invalid_argument);
}

} // namespace vkcompute::shader_profiling

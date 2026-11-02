// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>

#include <executorch/runtime/core/portable_type/half.h>

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

Json tensor(const Json& sizes, const std::string& dtype = "float32") {
  return {{"sizes", sizes}, {"dtype", dtype}, {"storage", "buffer"}};
}

template <typename T>
std::vector<uint8_t> bytes(const std::vector<T>& values) {
  std::vector<uint8_t> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}

Json linear_spec() {
  return {
      {"operator", "aten.linear.default"},
      {"args",
       {{{"tensor", "x"}},
        {{"tensor", "w"}},
        {{"tensor", "b"}},
        {{"tensor", "y"}}}},
      {"tensors",
       {{"x", tensor({2, 3})},
        {"w", tensor({2, 3})},
        {"b", tensor({2})},
        {"y", tensor({2, 2})}}}};
}

TEST(BenchmarkReference, LinearComputesBatchedProjectionWithBias) {
  const auto result = reference_outputs(
      linear_spec(),
      {},
      {{"x", bytes<float>({1, 2, 3, -1, 0, 2})},
       {"w", bytes<float>({2, -1, 1, 0, 3, -2})},
       {"b", bytes<float>({0.5, -0.5})}},
      2);
  EXPECT_EQ(result.at("y"), (std::vector<double>{3.5, -0.5, 0.5, -4.5}));
}

TEST(BenchmarkReference, LinearSupportsLeadingDimensionsAndNoBias) {
  Json spec = linear_spec();
  spec["tensors"]["x"]["sizes"] = {1, 2, 3};
  spec["tensors"]["y"]["sizes"] = {1, 2, 2};
  spec["args"][2] = nullptr;
  const auto result = reference_outputs(
      spec,
      {},
      {{"x", bytes<float>({1, 2, 3, -1, 0, 2})},
       {"w", bytes<float>({2, -1, 1, 0, 3, -2})}});
  EXPECT_EQ(result.at("y"), (std::vector<double>{3, 0, 0, -4}));
}

TEST(
    BenchmarkReference,
    LinearTilesAndTailsMatchScalarOracleAcrossThreadCounts) {
  constexpr size_t kM = 7;
  constexpr size_t kK = 17;
  constexpr size_t kN = 129;
  Json spec = linear_spec();
  spec["tensors"]["x"]["sizes"] = {kM, kK};
  spec["tensors"]["w"]["sizes"] = {kN, kK};
  spec["tensors"]["b"]["sizes"] = {kN};
  spec["tensors"]["y"]["sizes"] = {kM, kN};
  std::vector<float> input(kM * kK);
  std::vector<float> weight(kN * kK);
  std::vector<float> bias(kN);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = int(i % 7) - 3;
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    weight[i] = int(i % 11) - 5;
  }
  for (size_t i = 0; i < bias.size(); ++i) {
    bias[i] = int(i % 3) - 1;
  }
  std::vector<double> expected(kM * kN);
  for (size_t row = 0; row < kM; ++row) {
    for (size_t column = 0; column < kN; ++column) {
      double value = bias[column];
      for (size_t inner = 0; inner < kK; ++inner) {
        value += double(input[row * kK + inner]) *
            double(weight[column * kK + inner]);
      }
      expected[row * kN + column] = value;
    }
  }
  const std::map<std::string, std::vector<uint8_t>> data = {
      {"x", bytes(input)}, {"w", bytes(weight)}, {"b", bytes(bias)}};
  const auto single_thread = reference_outputs(spec, {}, data, 1);
  const auto eight_threads = reference_outputs(spec, {}, data, 8);
  EXPECT_EQ(single_thread.at("y"), expected);
  EXPECT_EQ(eight_threads, single_thread);
}

TEST(BenchmarkReference, HalfInputsAreRoundedAndPromotedInputsStayFloat) {
  Json spec = linear_spec();
  spec["args"][2] = nullptr;
  spec["tensors"]["x"] = tensor({1, 1}, "float16");
  spec["tensors"]["w"] = tensor({1, 1}, "float16");
  spec["tensors"]["w"]["storage"] = "constant";
  spec["tensors"]["y"] = tensor({1, 1}, "float16");
  using Half = executorch::runtime::etensor::Half;
  const std::map<std::string, std::vector<uint8_t>> data = {
      {"x", bytes<float>({1.0004f})}, {"w", bytes<Half>({Half(1.0f)})}};
  EXPECT_EQ(reference_outputs(spec, {}, data).at("y")[0], 1.0);
  EXPECT_EQ(
      reference_outputs(spec, {{"x", "float32"}, {"y", "float32"}}, data)
          .at("y")[0],
      double(1.0004f));
}

TEST(BenchmarkReference, LinearRejectsMismatchedOutputShape) {
  Json spec = linear_spec();
  spec["tensors"]["y"]["sizes"] = {4, 1};
  EXPECT_THROW(
      reference_outputs(
          spec,
          {},
          {{"x", bytes<float>({1, 2, 3, -1, 0, 2})},
           {"w", bytes<float>({2, -1, 1, 0, 3, -2})},
           {"b", bytes<float>({0, 0})}}),
      std::invalid_argument);
}

TEST(BenchmarkReference, GroupedConvolutionHonorsStridePaddingAndDilation) {
  const Json spec = {
      {"operator", "aten.convolution.default"},
      {"args",
       {{{"tensor", "x"}},
        {{"tensor", "w"}},
        {{"tensor", "b"}},
        {{"int_list", {1, 2}}},
        {{"int_list", {0, 1}}},
        {{"int_list", {1, 2}}},
        {{"bool", false}},
        {{"int_list", {0, 0}}},
        {{"int", 2}},
        {{"tensor", "y"}}}},
      {"tensors",
       {{"x", tensor({1, 2, 1, 5})},
        {"w", tensor({2, 1, 1, 2})},
        {"b", tensor({2})},
        {"y", tensor({1, 2, 1, 3})}}}};
  const auto result = reference_outputs(
      spec,
      {},
      {{"x", bytes<float>({1, 2, 3, 4, 5, 10, 20, 30, 40, 50})},
       {"w", bytes<float>({1, 2, -1, 1})},
       {"b", bytes<float>({1, -1})}},
      2);
  EXPECT_EQ(result.at("y"), (std::vector<double>{5, 11, 5, 19, 19, -41}));
}

TEST(BenchmarkReference, Q4LinearUnpacksNibblesAndUsesPerGroupScales) {
  const Json spec = {
      {"operator", "et_vk.q4gsw_linear.default"},
      {"args",
       {{{"tensor", "x"}},
        {{"tensor", "w"}},
        {{"tensor", "scales"}},
        {{"int", 2}},
        {{"tensor", "b"}},
        {{"tensor", "y"}}}},
      {"tensors",
       {{"x", tensor({1, 4})},
        {"w", tensor({2, 2}, "uint8")},
        {"scales", tensor({2, 2})},
        {"b", tensor({2})},
        {"y", tensor({1, 2})}}}};
  const auto result = reference_outputs(
      spec,
      {},
      {{"x", bytes<float>({1, 2, 3, 4})},
       {"w", bytes<uint8_t>({0x98, 0x76, 0xAF, 0x80})},
       {"scales", bytes<float>({1, 2, 3, 4})},
       {"b", bytes<float>({0.5, -0.5})}});
  EXPECT_EQ(result.at("y"), (std::vector<double>{-27.5, -74.5}));
}

TEST(BenchmarkReference, Q8ConvolutionAppliesZeroPointsScalesBiasAndClamping) {
  const Json spec = {
      {"operator", "et_vk.q8ta_conv2d.default"},
      {"args",
       {{{"tensor", "x"}},
        {{"float", 0.5}},
        {{"int", 2}},
        {{"tensor", "w"}},
        {{"tensor", "sums"}},
        {{"tensor", "scales"}},
        {{"float", 0.25}},
        {{"int", -3}},
        {{"tensor", "b"}},
        {{"int_list", {1, 1}}},
        {{"int_list", {1, 1}}},
        {{"int_list", {0, 0}}},
        {{"int_list", {1, 1}}},
        {{"int", 1}},
        {{"string", "none"}},
        {{"tensor", "y"}}}},
      {"tensors",
       {{"x", tensor({1, 2, 1, 2}, "int8x4")},
        {"w", tensor({2, 2}, "int8")},
        {"sums", tensor({2}, "int32")},
        {"scales", tensor({2})},
        {"b", tensor({2})},
        {"y", tensor({1, 2, 1, 2}, "int8x4")}}}};
  const auto result = reference_outputs(
      spec,
      {},
      {{"x", bytes<int8_t>({4, -2, 8, 3})},
       {"w", bytes<int8_t>({1, 2, -2, 1})},
       {"sums", bytes<int32_t>({3, -1})},
       {"scales", bytes<float>({0.25, 20})},
       {"b", bytes<float>({0.5, -1})}});
  EXPECT_EQ(result.at("y"), (std::vector<double>{6, -2, 73, 127}));
  Json relu = spec;
  relu["args"][14]["string"] = "relu";
  EXPECT_EQ(
      reference_outputs(
          relu,
          {},
          {{"x", bytes<int8_t>({0, 0, 0, 0})},
           {"w", bytes<int8_t>({1, 2, -2, 1})},
           {"sums", bytes<int32_t>({3, -1})},
           {"scales", bytes<float>({0.25, 20})},
           {"b", bytes<float>({0, 0})}})
          .at("y"),
      (std::vector<double>{-3, -3, 77, 77}));
}
TEST(BenchmarkReference, RawAdditionBroadcastsAndAppliesAlpha) {
  const Json spec = {
      {"shader", "binary_add_texture3d_float"},
      {"bindings",
       {{{"tensor", "out"}, {"access", "write"}},
        {{"tensor", "a"}},
        {{"tensor", "b"}}}},
      {"push_constants", {{{"dtype", "float32"}, {"data", {2}}}}},
      {"tensors",
       {{"out", tensor({2, 3})}, {"a", tensor({2, 1})}, {"b", tensor({3})}}}};
  EXPECT_EQ(
      reference_outputs(
          spec,
          {},
          {{"a", bytes<float>({1, 10})}, {"b", bytes<float>({2, 3, 4})}})
          .at("out"),
      (std::vector<double>{5, 7, 9, 14, 16, 18}));
}

TEST(BenchmarkReference, CopyPreservesLogicalValues) {
  const Json spec = {
      {"shader", "buffer_to_buffer_float"},
      {"bindings",
       {{{"tensor", "out"}, {"access", "write"}}, {{"tensor", "in"}}}},
      {"tensors", {{"out", tensor({4})}, {"in", tensor({4})}}}};
  EXPECT_EQ(
      reference_outputs(spec, {}, {{"in", bytes<float>({-1, 0, 2, 4})}})
          .at("out"),
      (std::vector<double>{-1, 0, 2, 4}));
}

TEST(BenchmarkReference, ExplicitReferenceOverridesUnknownShader) {
  const Json spec = {{"shader", "custom"}, {"reference", "copy"}};
  EXPECT_EQ(reference_name(spec), "copy");
  EXPECT_THROW(reference_name({{"shader", "custom"}}), std::invalid_argument);
}

TEST(BenchmarkReference, ComparisonReportsBoundedMismatchSummary) {
  Json spec = linear_spec();
  spec["validation_method"] = {
      {"function", "is_close"}, {"atol", 0.1}, {"rtol", 0.01}};
  const Report result = compare_outputs(
      spec, {{"y", {1, 2, 3, 4}}}, {{"y", {1.05, 2.5, 3, 4.5}}}, {});
  EXPECT_EQ(result.at("status"), "FAIL");
  EXPECT_EQ(result.at("outputs").at("y").at("mismatch_count"), 2);
  EXPECT_DOUBLE_EQ(result.at("outputs").at("y").at("max_abs_error"), 0.5);
  EXPECT_FALSE(result.at("outputs").at("y").contains("values"));
  EXPECT_EQ(
      compare_outputs(spec, {{"y", {1, 2, 3, 4}}}, {{"y", {1.05, 2, 3, 4}}}, {})
          .at("status"),
      "PASS");
}

TEST(BenchmarkReference, SelectedOutputsDoNotHideMissingReferenceResults) {
  Json spec = linear_spec();
  const ReferenceOutputs expected{{"y", {1, 2, 3, 4}}, {"b", {1, 2}}};
  const ReferenceOutputs actual{{"y", {1, 2, 3, 4}}, {"b", {10, 20}}};
  EXPECT_EQ(compare_outputs(spec, expected, actual, {}).at("status"), "FAIL");
  spec["validation_method"] = {{"function", "is_close"}, {"outputs", {"y"}}};
  const Report selected = compare_outputs(spec, expected, actual, {});
  EXPECT_EQ(selected.at("status"), "PASS");
  EXPECT_EQ(selected.at("outputs").size(), 1);
  EXPECT_EQ(
      compare_outputs(spec, expected, {{"b", {1, 2}}}, {}).at("status"),
      "FAIL");
  EXPECT_EQ(
      compare_outputs(spec, {{"b", {1, 2}}}, actual, {}).at("status"), "FAIL");
}

TEST(BenchmarkReference, ComparisonRejectsMissingShortAndUnexpectedOutputs) {
  const Json spec = linear_spec();
  const ReferenceOutputs expected = {{"y", {1, 2, 3, 4}}};
  EXPECT_EQ(compare_outputs(spec, expected, {}, {}).at("status"), "FAIL");
  EXPECT_EQ(
      compare_outputs(spec, expected, {{"y", {1}}}, {}).at("status"), "FAIL");
  EXPECT_EQ(
      compare_outputs(spec, expected, {{"y", {1, 2, 3, 4}}, {"extra", {1}}}, {})
          .at("status"),
      "FAIL");
}

TEST(BenchmarkReference, ComparisonRejectsNonfiniteValues) {
  const Json spec = linear_spec();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(
      compare_outputs(
          spec, {{"y", {1, 2, 3, 4}}}, {{"y", {1, nan, inf, 4}}}, {})
          .at("status"),
      "FAIL");
  EXPECT_EQ(
      compare_outputs(
          spec, {{"y", {nan, 2, 3, 4}}}, {{"y", {nan, 2, 3, 4}}}, {})
          .at("status"),
      "FAIL");
}

TEST(BenchmarkReference, ComparisonDefaultsDependOnActualDtype) {
  Json spec = linear_spec();
  spec["tensors"]["y"]["dtype"] = "float16";
  const ReferenceOutputs expected = {{"y", {1, 2, 3, 4}}};
  const ReferenceOutputs close = {{"y", {1.005, 2, 3, 4}}};
  EXPECT_EQ(compare_outputs(spec, expected, close, {}).at("status"), "PASS");
  EXPECT_EQ(
      compare_outputs(spec, expected, close, {{"y", "float32"}}).at("status"),
      "FAIL");
  EXPECT_EQ(
      compare_outputs(spec, expected, close, {{"y", "int8x4"}}).at("status"),
      "FAIL");
  spec["validation_method"] = {{"function", "is_close"}, {"atol", -1}};
  EXPECT_THROW(
      compare_outputs(spec, expected, close, {}), std::invalid_argument);
}

} // namespace
} // namespace vkcompute::shader_profiling

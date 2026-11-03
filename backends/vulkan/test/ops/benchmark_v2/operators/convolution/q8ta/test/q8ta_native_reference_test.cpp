// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taConv2dReference.h>

#include <gtest/gtest.h>

#include <cmath>

namespace vkcompute::shader_profiling {
namespace {
using reference::Tensor;
using reference::TensorMap;

Json native_spec() {
  return {
      {"operator", "et_vk.q8ta_conv2d.default"},
      {"args",
       {{{"tensor", "x"}},
        {{"float", 1.0}},
        {{"int", 0}},
        {{"tensor", "w"}},
        {{"tensor", "sums"}},
        {{"tensor", "scales"}},
        {{"float", 1.0}},
        {{"int", 0}},
        {{"tensor", "bias"}},
        {{"int_list", {1, 1}}},
        {{"int_list", {1, 1}}},
        {{"int_list", {0, 0}}},
        {{"int_list", {1, 1}}},
        {{"int", 1}},
        {{"string", "none"}},
        {{"tensor", "y"}}}},
      {"tensors",
       {{"y",
         {{"sizes", {1, 1, 1, 1}},
          {"dtype", "int8x4"},
          {"storage", "buffer"},
          {"layout", "4c1w"}}}}}};
}

TEST(Q8taNativeReference, RegistersNativeOperatorsOnly) {
  for (const char* name :
       {"et_vk.q8ta_conv2d.default",
        "et_vk.q8ta_conv2d_general.default",
        "et_vk.q8ta_conv2d_im2col.default",
        "et_vk.q8ta_conv2d_dw.default",
        "et_vk.q8ta_conv2d_pw.default"})
    EXPECT_EQ(reference_registry().operator_reference(name), "q8ta_conv2d");
  EXPECT_EQ(
      reference_registry().operator_reference(
          "et_vk.q8ta_conv2d_transposed.default"),
      "q8ta_transposed");
  EXPECT_THROW(
      reference_registry().operator_reference(
          "benchmark_v2.q8ta_conv2d.default"),
      std::invalid_argument);
}

TEST(
    Q8taNativeReference,
    QuantizedInputOutputHonorZeroPointsBiasAndPaddedRows) {
  Json spec = native_spec();
  spec["args"][1]["float"] = .5;
  spec["args"][2]["int"] = 2;
  spec["args"][6]["float"] = .5;
  spec["args"][7]["int"] = -3;
  const TensorMap data{
      {"x", {{1, 1, 1, 1}, {5}}},
      {"w", {{1, 4}, {2, 0, 0, 0}}},
      {"sums", {{1}, {2}}},
      {"scales", {{1}, {.25}}},
      {"bias", {{1}, {.25}}}};
  EXPECT_EQ(
      reference::q8ta_conv2d_reference(spec, data, 1).values,
      (std::vector<double>{-1}));
}

TEST(Q8taNativeReference, DepthwiseHwcWeightsPreserveChannelAndKernelOrder) {
  Json spec = native_spec();
  spec["operator"] = "et_vk.q8ta_conv2d_dw.default";
  spec["args"][8] = nullptr;
  spec["args"][9]["int_list"] = {1, 2};
  spec["args"][13]["int"] = 2;
  spec["tensors"]["y"]["sizes"] = {1, 2, 1, 2};
  const TensorMap data{
      {"x", {{1, 2, 1, 3}, {1, 2, 3, 10, 20, 30}}},
      {"w", {{1, 2, 2}, {1, 4, 2, -1}}},
      {"sums", {{2}, {3, 3}}},
      {"scales", {{2}, {1, 1}}}};
  const auto one = reference::q8ta_conv2d_reference(spec, data, 1);
  EXPECT_EQ(one.values, (std::vector<double>{5, 8, 20, 50}));
  EXPECT_EQ(reference::q8ta_conv2d_reference(spec, data, 4).values, one.values);
}

TEST(Q8taNativeReference, TransposedOutputPaddingUsesNativeArgumentPositions) {
  Json spec = native_spec();
  spec["operator"] = "et_vk.q8ta_conv2d_transposed.default";
  spec["args"][8] = nullptr;
  spec["args"][9]["int_list"] = {1, 2};
  spec["args"][10]["int_list"] = {1, 2};
  spec["args"].insert(spec["args"].begin() + 12, Json{{"int_list", {0, 1}}});
  spec["tensors"]["y"]["sizes"] = {1, 1, 1, 5};
  const TensorMap data{
      {"x", {{1, 1, 1, 2}, {1, 2}}},
      {"w", {{1, 4}, {1, 2, 0, 0}}},
      {"sums", {{1}, {3}}},
      {"scales", {{1}, {1}}}};
  EXPECT_EQ(
      reference::q8ta_transposed_reference(spec, data, 2).values,
      (std::vector<double>{1, 2, 2, 4, 0}));
}

TEST(Q8taNativeReference, RequantizesNearestEvenAndSaturatesWithRelu) {
  Json spec = native_spec();
  spec["args"][1]["float"] = .5;
  spec["args"][8] = nullptr;
  spec["tensors"]["y"]["sizes"] = {1, 1, 1, 6};
  const TensorMap data{
      {"x", {{1, 1, 1, 6}, {1, 3, -1, -3, 127, -128}}},
      {"w", {{1, 4}, {1, 0, 0, 0}}},
      {"sums", {{1}, {1}}},
      {"scales", {{1}, {1}}}};
  EXPECT_EQ(
      reference::q8ta_conv2d_reference(spec, data, 1).values,
      (std::vector<double>{0, 2, 0, -2, 64, -64}));
  spec["args"][1]["float"] = 4;
  spec["args"][14]["string"] = "relu";
  EXPECT_EQ(
      reference::q8ta_conv2d_reference(spec, data, 1).values,
      (std::vector<double>{4, 12, 0, 0, 127, 0}));
}

TEST(Q8taNativeReference, ActualShaderSelectsIntegerCorrectionAndFusedBias) {
  Json spec = native_spec();
  Tensor input{{1, 1042, 1, 1}, std::vector<double>(1042, 127)};
  Tensor weight{{1, 1044}, std::vector<double>(1044, 127)};
  input.values[1041] = 1;
  weight.values[1040] = 24;
  weight.values[1041] = 7;
  weight.values[1042] = weight.values[1043] = 0;
  const TensorMap data{
      {"x", input},
      {"w", weight},
      {"sums", {{1}, {132111}}},
      {"scales", {{1}, {std::nextafter(1.0f, 2.0f)}}},
      {"bias", {{1}, {-16777216}}}};
  spec["reference_context"]["shaders"] = {"q8ta_conv2d_float"};
  EXPECT_EQ(reference::q8ta_conv2d_reference(spec, data, 1).values[0], 0);
  spec["reference_context"]["shaders"] = {"q8ta_conv2d_pw_float"};
  EXPECT_EQ(reference::q8ta_conv2d_reference(spec, data, 1).values[0], 1);
}
} // namespace
} // namespace vkcompute::shader_profiling

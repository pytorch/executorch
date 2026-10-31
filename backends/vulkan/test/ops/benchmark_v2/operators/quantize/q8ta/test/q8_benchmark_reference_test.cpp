// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Config.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Reference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantize/q8ta/Q8taReferenceCommon.h>
#include <gtest/gtest.h>
namespace vkcompute::shader_profiling {
namespace {
Json tensor_arg(
    const char* name,
    const Json& sizes,
    const Json& data,
    const char* dtype = "float32",
    const char* layout = "width") {
  return {
      {"tensor",
       {{"name", name},
        {"sizes", sizes},
        {"dtype", dtype},
        {"layout", layout},
        {"storage",
         std::string(name) == "input" || std::string(name) == "output" ||
                 std::string(name) == "other"
             ? "buffer"
             : "constant"},
        {"data", data}}}};
}
std::vector<double> compute(const char* op, const Json& args) {
  const Json spec =
      parse_configs({{"operator", op}, {"args_sweep", Json::array({args})}})
          .front();
  auto data = make_tensor_data(spec.at("tensors"));
  data.erase("output");
  return reference_outputs(spec, {}, data, 2).at("output");
}
TEST(Q8BenchmarkReference, AddBroadcastsQuantizedInputsAndHonorsAlpha) {
  const Json args = Json::array(
      {tensor_arg("input", {2, 1}, {2, 4}, "int8x4", "4w"),
       tensor_arg("other", {2}, {-2, 4}, "int8x4", "4w"),
       {{"float", .5}},
       {{"int", 0}},
       {{"float", .25}},
       {{"int", 0}},
       {{"float", .5}},
       {{"int", 0}},
       {{"float", 2}},
       tensor_arg("output", {2, 2}, {{"fill", 0}}, "int8x4", "4w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_add.default", args),
      (std::vector<double>{0, 6, 2, 8}));
}
TEST(Q8BenchmarkReference, AddAlphaAndSaturationWithBroadcasting) {
  Json args = Json::array(
      {tensor_arg("input", {2, 1}, {120, -120}, "int8x4", "4w"),
       tensor_arg("other", {2}, {-20, 20}, "int8x4", "4w"),
       {{"float", 1}},
       {{"int", 0}},
       {{"float", 1}},
       {{"int", 0}},
       {{"float", 1}},
       {{"int", 0}},
       {{"float", 0}},
       tensor_arg("output", {2, 2}, {{"fill", 0}}, "int8x4", "4w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_add.default", args),
      (std::vector<double>{120, 120, -120, -120}));
  args[8]["float"] = .5;
  EXPECT_EQ(
      compute("et_vk.q8ta_add.default", args),
      (std::vector<double>{110, 127, -128, -110}));
  args[8]["float"] = -1;
  EXPECT_EQ(
      compute("et_vk.q8ta_add.default", args),
      (std::vector<double>{127, 100, -100, -128}));
}
TEST(Q8BenchmarkReference, ClonePreservesSignedBytesAcrossLayouts) {
  const Json args = Json::array(
      {tensor_arg("input", {5}, {-128, -1, 0, 1, 127}, "int8x4", "4w"),
       nullptr,
       tensor_arg("output", {5}, {{"fill", 0}}, "int8x4", "4c1w")});
  EXPECT_EQ(
      compute("aten.clone.default", args),
      (std::vector<double>{-128, -1, 0, 1, 127}));
}
TEST(Q8BenchmarkReference, QuantizeRoundsAndClampsIntegerOutput) {
  const Json args = Json::array(
      {tensor_arg("input", {5}, {-100, -.26, 0, .26, 100}),
       {{"float", .5}},
       {{"int", 0}},
       {{"int", -128}},
       {{"int", 127}},
       {{"int", 2}},
       tensor_arg("output", {5}, {{"fill", 0}}, "int8x4", "4w")});
  EXPECT_EQ(
      compute("quantized_decomposed.quantize_per_tensor.default", args),
      (std::vector<double>{-128, -1, 0, 1, 127}));
}
TEST(Q8BenchmarkReference, DequantizeAppliesScaleAndZeroPoint) {
  const Json args = Json::array(
      {tensor_arg("input", {5}, {-128, -1, 0, 1, 127}, "int8x4", "4w"),
       {{"float", .5}},
       {{"int", -2}},
       {{"int", -128}},
       {{"int", 127}},
       {{"int", 2}},
       {{"int", 10}},
       tensor_arg("output", {5}, {{"fill", 0}})});
  EXPECT_EQ(
      compute("quantized_decomposed.dequantize_per_tensor.default", args),
      (std::vector<double>{-63, .5, 1, 1.5, 64.5}));
}
TEST(Q8BenchmarkReference, ReluUsesQuantizedZeroPoints) {
  const Json args = Json::array(
      {tensor_arg("input", {4}, {-4, -2, 0, 2}, "int8x4", "4w"),
       {{"float", 1}},
       {{"int", -2}},
       {{"float", .5}},
       {{"int", 3}},
       tensor_arg("output", {4}, {{"fill", 0}}, "int8x4", "4w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_relu.default", args),
      (std::vector<double>{3, 3, 7, 11}));
}
TEST(Q8BenchmarkReference, PixelShufflePermutesQuantizedBytes) {
  const Json args = Json::array(
      {tensor_arg("input", {1, 4, 1, 1}, {1, 2, 3, 4}, "int8x4", "4w4c"),
       {{"float", 1}},
       {{"int", 0}},
       {{"float", 1}},
       {{"int", 0}},
       {{"int", 2}},
       tensor_arg("output", {1, 1, 2, 2}, {{"fill", 0}}, "int8x4", "4c1w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_pixel_shuffle.default", args),
      (std::vector<double>{1, 2, 3, 4}));
}
TEST(Q8BenchmarkReference, PixelShuffleRequantizesUsingInverseOutputScale) {
  const Json args = Json::array(
      {tensor_arg("input", {1, 4, 1, 1}, {-128, -1, 1, 127}, "int8x4", "4w4c"),
       {{"float", .5}},
       {{"int", -1}},
       {{"float", 2}},
       {{"int", 3}},
       {{"int", 2}},
       tensor_arg("output", {1, 1, 2, 2}, {{"fill", 0}}, "int8x4", "4c1w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_pixel_shuffle.default", args),
      (std::vector<double>{-124, 3, 5, 127}));
}
Json convolution_args(
    const Json& input,
    const Json& weight,
    const Json& scales,
    const Json& bias,
    const Json& output) {
  return Json::array(
      {input,
       {{"float", 1}},
       {{"int", 0}},
       weight,
       tensor_arg("sums", {4}, {0, 0, 0, 0}, "int32"),
       scales,
       {{"float", 1}},
       {{"int", 0}},
       bias,
       {{"int_list", {1, 1}}},
       {{"int_list", {1, 1}}},
       {{"int_list", {0, 0}}},
       {{"int_list", {1, 1}}},
       {{"int", 1}},
       {{"string", "none"}},
       output});
}
TEST(Q8BenchmarkReference, ConvolutionIgnoresPaddedWeightAndChannelParameters) {
  const Json args = convolution_args(
      tensor_arg("input", {1, 3, 1, 1}, {1, 2, 3}, "int8x4", "4c"),
      tensor_arg("weight", {2, 4}, {1, 2, 3, 99, 4, 5, 6, -99}, "int8"),
      tensor_arg("scales", {4}, {1, 1, 99, 99}),
      tensor_arg("bias", {4}, {0, 0, 99, 99}),
      tensor_arg("output", {1, 2, 1, 1}, {{"fill", 0}}, "int8x4", "4c"));
  EXPECT_EQ(
      compute("et_vk.q8ta_conv2d.default", args),
      (std::vector<double>{14, 32}));
}
TEST(Q8BenchmarkReference, DepthwiseReadsHeightWidthChannelWeightOrder) {
  Json args = convolution_args(
      tensor_arg("input", {1, 2, 1, 2}, {1, 2, 3, 4}, "int8x4", "4c"),
      tensor_arg("weight", {1, 1, 2}, {2, -1}, "int8"),
      tensor_arg("scales", {2}, {1, 1}),
      tensor_arg("bias", {2}, {0, 0}),
      tensor_arg("output", {1, 2, 1, 2}, {{"fill", 0}}, "int8x4", "4c"));
  args[13]["int"] = 2;
  EXPECT_EQ(
      compute("et_vk.q8ta_conv2d_dw.default", args),
      (std::vector<double>{2, 4, -3, -4}));
}
TEST(Q8BenchmarkReference, LinearUsesIntegerInputsAndWeightSumCorrection) {
  const Json args = Json::array(
      {tensor_arg("input", {1, 4}, {4, 2, 3, 1}, "int8x4", "4h4w"),
       {{"float", 1}},
       {{"int", 1}},
       tensor_arg("weight", {2, 4}, {1, -1, 0, 0, 2, 0, -1, 0}, "int8"),
       tensor_arg("sums", {2}, {0, 1}, "int32"),
       tensor_arg("scales", {2}, {1, 1}),
       {{"float", 1}},
       {{"int", 0}},
       tensor_arg("bias", {2}, {1, -1}),
       {{"string", "none"}},
       tensor_arg("output", {1, 2}, {{"fill", 0}}, "int8x4", "4h4w")});
  EXPECT_EQ(
      compute("et_vk.q8ta_linear.default", args), (std::vector<double>{3, 3}));
}
TEST(Q8BenchmarkReference, TransposedConvolutionExpandsStridedKernel) {
  Json args = convolution_args(
      tensor_arg("input", {1, 1, 1, 1}, {2}, "int8x4", "4c1w"),
      tensor_arg("weight", {1, 4}, {1, 2, 3, 4}, "int8"),
      tensor_arg("scales", {1}, {1}),
      nullptr,
      tensor_arg("output", {1, 1, 2, 2}, {{"fill", 0}}, "int8x4", "4c1w"));
  args[9]["int_list"] = {2, 2};
  args[10]["int_list"] = {2, 2};
  args.insert(args.begin() + 12, Json{{"int_list", {0, 0}}});
  EXPECT_EQ(
      compute("et_vk.q8ta_conv2d_transposed.default", args),
      (std::vector<double>{2, 4, 6, 8}));
}
TEST(
    Q8BenchmarkReference,
    AccumulatorRequantizationHandlesBiasReluAndClamping) {
  EXPECT_EQ(
      reference::q8ta::requantize_accumulator(7, .5f, 1.0f, 1.0f, -3, .5f), 1);
  EXPECT_EQ(
      reference::q8ta::requantize_accumulator(
          -8, .5f, 1.0f, 1.0f, 4, 0.0f, true),
      4);
  EXPECT_EQ(
      reference::q8ta::requantize_accumulator(10000, 1.0f, 1.0f, 1.0f, 0), 127);
  EXPECT_EQ(
      reference::q8ta::requantize_accumulator(-10000, 1.0f, 1.0f, 1.0f, 0),
      -128);
}
TEST(
    Q8BenchmarkReference,
    NativeLinearThroughputCountsIntegerTensorTrafficAndMacs) {
  const Json args = Json::array(
      {tensor_arg("input", {1, 4}, {1, 2, 3, 4}, "int8x4", "4h4w"),
       {{"float", 1}},
       {{"int", 0}},
       tensor_arg("weight", {2, 4}, {1, 1, 1, 1, 1, 1, 1, 1}, "int8"),
       tensor_arg("sums", {2}, {4, 4}, "int32"),
       tensor_arg("scales", {2}, {1, 1}),
       {{"float", 1}},
       {{"int", 0}},
       nullptr,
       {{"string", "none"}},
       tensor_arg("output", {1, 2}, {{"fill", 0}}, "int8x4", "4h4w")});
  const Json spec = parse_configs({{"operator", "et_vk.q8ta_linear.default"},
                                   {"args_sweep", Json::array({args})}})
                        .front();
  const auto work = estimate_work(
      Json(operator_report(spec, Json::object(), {{"output", "write"}})),
      true,
      true);
  ASSERT_TRUE(work.memory_bytes.has_value());
  ASSERT_TRUE(work.compute_flops.has_value());
  EXPECT_EQ(*work.memory_bytes, 30);
  EXPECT_EQ(*work.compute_flops, 16);
}
TEST(
    Q8BenchmarkReference,
    QuantizationUsesReciprocalAndRoundsTiesBeforeZeroPoint) {
  EXPECT_EQ(
      reference::q8ta::quantize(-0.17499999702f, 0.070000000298f, -3), -5);
  EXPECT_EQ(reference::q8ta::quantize(0.17499999702f, 0.070000000298f, 3), 5);
  EXPECT_EQ(
      reference::q8ta::quantize(-0.324000000954f, 0.00800000037998f, -2), -42);
  const std::vector<float> values{
      .5f, 1.5f, 2.5f, 3.5f, -.5f, -1.5f, -2.5f, -3.5f};
  const std::vector<int64_t> expected{3, 5, 5, 7, 3, 1, 1, -1};
  for (size_t i = 0; i < values.size(); ++i)
    EXPECT_EQ(reference::q8ta::quantize(values[i], 1.0f, 3), expected[i]);
}
TEST(
    Q8BenchmarkReference,
    ActivationQuantizedConvolutionAppliesScaleAfterIntegerDot) {
  const float scale = .07f, weight_scale = .013f;
  const Json args = Json::array(
      {tensor_arg("input", {1, 1, 1, 1}, {-0.17499999702f}),
       {{"float", scale}},
       {{"int", -3}},
       tensor_arg("weight", {1, 4}, {127, 0, 0, 0}, "int8"),
       tensor_arg("sums", {1}, {127}, "int32"),
       tensor_arg("scales", {1}, {weight_scale}),
       tensor_arg("bias", {1}, {.5}),
       {{"int_list", {1, 1}}},
       {{"int_list", {1, 1}}},
       {{"int_list", {0, 0}}},
       {{"int_list", {1, 1}}},
       {{"int", 1}},
       tensor_arg("output", {1, 1, 1, 1}, {{"fill", 0}})});
  const float expected = float(-2 * 127) * (scale * weight_scale) + .5f;
  EXPECT_EQ(
      compute("et_vk.conv2d_q8ta_q8csw.default", args),
      (std::vector<double>{expected}));
}
TEST(
    Q8BenchmarkReference,
    ActivationQuantizedLinearCancelsIntegersBeforeScaling) {
  constexpr int64_t k = 2048;
  Json values = Json::array_t(size_t(k), Json(.07f));
  Json weights = Json::array_t(size_t(k), Json(-127));
  for (size_t i = 0; i < size_t(k) / 2; ++i)
    weights[i] = 127;
  const Json args = Json::array(
      {tensor_arg("input", {1, k}, values),
       {{"float", .07f}},
       {{"int", 0}},
       tensor_arg("weight", {1, k}, weights, "int8"),
       tensor_arg("sums", {1}, {0}, "int32"),
       tensor_arg("scales", {1}, {.013f}),
       tensor_arg("bias", {1}, {.5}),
       tensor_arg("output", {1, 1}, {{"fill", 0}})});
  EXPECT_EQ(
      compute("et_vk.linear_q8ta_q8csw.default", args),
      (std::vector<double>{.5}));
}
} // namespace
} // namespace vkcompute::shader_profiling

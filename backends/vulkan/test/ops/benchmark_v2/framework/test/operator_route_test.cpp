// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/api_abstractions/VulkanTest.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/RouteRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/Conv2dRoutes.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/q8ta/Q8taConv2dRoutes.h>

#include <executorch/runtime/core/portable_type/half.h>

#include <gtest/gtest.h>

#include <stdexcept>

namespace vkcompute::shader_profiling {
using BenchmarkOperatorRoute = VulkanTest;

namespace {

struct FpConvCase {
  std::vector<float> weights;
  std::vector<float> bias;
  std::vector<executorch::runtime::etensor::Half> half_weights;
  std::vector<executorch::runtime::etensor::Half> half_bias;
  ComputeGraph graph{GraphConfig{}};
  std::vector<ValueRef> args;

  explicit FpConvCase(
      vkapi::ScalarType dtype = vkapi::kFloat,
      utils::StorageType storage = utils::kTexture3D,
      utils::GPUMemoryLayout layout = utils::kChannelsPacked,
      int64_t batch = 1,
      int64_t channels = 4,
      int64_t groups = 1)
      : weights(8 * (channels / groups) * 9, 0.25f),
        bias(8, 0.1f),
        half_weights(weights.size(), executorch::runtime::etensor::Half(0.25f)),
        half_bias(bias.size(), executorch::runtime::etensor::Half(0.1f)) {
    const void* weight_data = dtype == vkapi::kHalf
        ? static_cast<const void*>(half_weights.data())
        : weights.data();
    const void* bias_data = dtype == vkapi::kHalf
        ? static_cast<const void*>(half_bias.data())
        : bias.data();
    args = {
        graph.add_tensor({batch, channels, 7, 7}, dtype, storage, layout),
        graph.add_tensorref({8, channels / groups, 3, 3}, dtype, weight_data),
        graph.add_tensorref({8}, dtype, bias_data),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar(false),
        graph.add_scalar_list(std::vector<int64_t>{0, 0}),
        graph.add_scalar(groups),
        graph.add_tensor({batch, 8, 7, 7}, dtype, storage, layout)};
  }
};

struct Q8ConvCase {
  std::vector<int8_t> weights;
  std::vector<int32_t> sums;
  std::vector<float> scales;
  std::vector<float> bias;
  ComputeGraph graph{GraphConfig{}};
  std::vector<ValueRef> args;

  explicit Q8ConvCase(
      int64_t batch = 1,
      int64_t channels = 4,
      int64_t groups = 1)
      : weights(8 * (channels / groups) * 9, 1),
        sums(8, int32_t((channels / groups) * 9)),
        scales(8, 0.1f),
        bias(8, 0.0f) {
    args = {
        graph.add_tensor(
            {batch, channels, 7, 7},
            vkapi::kInt8x4,
            utils::kBuffer,
            utils::kPackedInt8_4W4C),
        graph.add_scalar(0.25),
        graph.add_scalar(int64_t(0)),
        graph.add_tensorref(
            {8, (channels / groups) * 9}, vkapi::kChar, weights.data()),
        graph.add_tensorref({8}, vkapi::kInt, sums.data()),
        graph.add_tensorref({8}, vkapi::kFloat, scales.data()),
        graph.add_scalar(0.25),
        graph.add_scalar(int64_t(0)),
        graph.add_tensorref({8}, vkapi::kFloat, bias.data()),
        graph.add_scalar_list(std::vector<int64_t>{3, 3}),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar_list(std::vector<int64_t>{1, 1}),
        graph.add_scalar(groups),
        graph.add_string("none"),
        graph.add_tensor(
            {batch, 8, 7, 7},
            vkapi::kInt8x4,
            utils::kBuffer,
            utils::kPackedInt8_4W4C)};
  }
};

TEST_F(
    BenchmarkOperatorRoute,
    DirectConvolutionBuildsProductionShaderForBothDtypes) {
  for (const auto dtype : {vkapi::kFloat, vkapi::kHalf}) {
    FpConvCase test(dtype);
    const Json selected =
        build_conv2d_direct(test.graph, test.args, Json::object());
    EXPECT_EQ(selected.at("name"), "direct");
    EXPECT_TRUE(selected.at("options").empty());
    ASSERT_EQ(test.graph.execute_nodes().size(), 1);
    EXPECT_EQ(test.graph.execute_nodes()[0]->name().find("conv2d_"), 0);
    EXPECT_EQ(
        inspect_conv2d_route(test.graph, test.args, 0, 1).at("name"), "direct");
  }
}

TEST_F(
    BenchmarkOperatorRoute,
    Im2colConvolutionHonorsAllIntermediateStorageOverrides) {
  for (const char* storage : {"buffer", "texture2d", "texture3d"}) {
    FpConvCase test;
    const Json selected = build_conv2d_im2col(
        test.graph, test.args, {{"intermediate_storage", storage}});
    EXPECT_EQ(selected.at("name"), "im2col");
    EXPECT_EQ(selected.at("options").at("intermediate_storage"), storage);
    ASSERT_EQ(test.graph.execute_nodes().size(), 2);
    EXPECT_EQ(
        test.graph.execute_nodes()[0]->name().find(
            std::string("conv2d_im2col_") + storage + "_"),
        0);
    EXPECT_EQ(
        test.graph.execute_nodes()[1]->name().find(
            std::string("conv2d_gemm_") + storage + "_"),
        0);
    EXPECT_EQ(inspect_conv2d_route(test.graph, test.args, 0, 2), selected);
  }
}

TEST_F(
    BenchmarkOperatorRoute,
    ForcedIm2colAllowsDilatedCasesExcludedByAutoPolicy) {
  FpConvCase test;
  test.args[4] = test.graph.add_scalar_list(std::vector<int64_t>{2, 2});
  test.args[5] = test.graph.add_scalar_list(std::vector<int64_t>{2, 2});
  EXPECT_EQ(
      build_conv2d_im2col(
          test.graph, test.args, {{"intermediate_storage", "buffer"}})
          .at("name"),
      "im2col");
  EXPECT_EQ(test.graph.execute_nodes().size(), 2);
}

TEST_F(
    BenchmarkOperatorRoute,
    UnsupportedConvolutionRoutesRejectBeforeAddingNodes) {
  FpConvCase batched(
      vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, 2);
  EXPECT_THROW(
      build_conv2d_direct(batched.graph, batched.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_conv2d_im2col(batched.graph, batched.args, Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(batched.graph.execute_nodes().empty());
  EXPECT_TRUE(batched.graph.prepack_nodes().empty());
  FpConvCase buffer(vkapi::kFloat, utils::kBuffer);
  EXPECT_THROW(
      build_conv2d_direct(buffer.graph, buffer.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_conv2d_im2col(buffer.graph, buffer.args, Json::object()),
      UnsupportedRoute);
  FpConvCase width(vkapi::kFloat, utils::kTexture3D, utils::kWidthPacked);
  EXPECT_THROW(
      build_conv2d_direct(width.graph, width.args, Json::object()),
      UnsupportedRoute);
  FpConvCase grouped(
      vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, 1, 8, 2);
  EXPECT_THROW(
      build_conv2d_im2col(grouped.graph, grouped.args, Json::object()),
      UnsupportedRoute);
  FpConvCase transposed;
  transposed.args[6] = transposed.graph.add_scalar(true);
  EXPECT_THROW(
      build_conv2d_im2col(transposed.graph, transposed.args, Json::object()),
      UnsupportedRoute);
}

TEST_F(BenchmarkOperatorRoute, ForcedDirectSupportsAlignedGroupedConvolution) {
  FpConvCase test(
      vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, 1, 8, 2);
  EXPECT_EQ(
      build_conv2d_direct(test.graph, test.args, Json::object()).at("name"),
      "direct");
  EXPECT_EQ(test.graph.execute_nodes().size(), 1);
}

TEST_F(BenchmarkOperatorRoute, UnknownOptionsAndGpuWeightsAreRejected) {
  FpConvCase test;
  EXPECT_THROW(
      build_conv2d_direct(
          test.graph, test.args, {{"intermediate_storage", "buffer"}}),
      std::invalid_argument);
  EXPECT_THROW(
      build_conv2d_im2col(test.graph, test.args, {{"unknown", true}}),
      std::invalid_argument);
  EXPECT_THROW(
      build_conv2d_im2col(
          test.graph, test.args, {{"intermediate_storage", "unknown"}}),
      std::invalid_argument);
  test.args[1] = test.graph.add_tensor(
      {8, 4, 3, 3}, vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked);
  EXPECT_THROW(
      build_conv2d_direct(test.graph, test.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_conv2d_im2col(test.graph, test.args, Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(test.graph.execute_nodes().empty());
}

TEST_F(
    BenchmarkOperatorRoute,
    Q8RoutesBuildNativeGeneralAndIm2colWithoutPerfPolicy) {
  for (int64_t batch : {int64_t(1), int64_t(2)}) {
    Q8ConvCase direct(batch);
    EXPECT_EQ(
        build_q8ta_conv2d_direct(direct.graph, direct.args, Json::object())
            .at("name"),
        "direct");
    ASSERT_EQ(direct.graph.execute_nodes().size(), 1);
    EXPECT_EQ(direct.graph.execute_nodes()[0]->name().find("q8ta_conv2d"), 0);
    Q8ConvCase im2col(batch);
    EXPECT_EQ(
        build_q8ta_conv2d_im2col(im2col.graph, im2col.args, Json::object())
            .at("name"),
        "im2col");
    ASSERT_EQ(im2col.graph.execute_nodes().size(), 2);
    EXPECT_EQ(im2col.graph.execute_nodes()[0]->name(), "q8ta_im2col");
    EXPECT_EQ(
        inspect_q8ta_conv2d_route(im2col.graph, im2col.args, 0, 2).at("name"),
        "im2col");
  }
}

TEST_F(BenchmarkOperatorRoute, Q8Im2colRejectsPackingAndUnsupportedOptions) {
  Q8ConvCase test;
  test.args[0] = test.graph.add_tensor(
      {1, 4, 7, 7}, vkapi::kInt8x4, utils::kBuffer, utils::kPackedInt8_4W);
  EXPECT_THROW(
      build_q8ta_conv2d_im2col(test.graph, test.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_q8ta_conv2d_direct(test.graph, test.args, {{"unknown", true}}),
      std::invalid_argument);
  EXPECT_TRUE(test.graph.execute_nodes().empty());
}

TEST_F(
    BenchmarkOperatorRoute,
    RegistryPreservesArgumentsAndReportsRequestedAndResolvedRoutes) {
  FpConvCase explicit_case;
  const auto original_args = explicit_case.args;
  const Json requested = {
      {"name", "im2col"}, {"options", {{"intermediate_storage", "texture2d"}}}};
  const Json result = build_operator_route(
      {{"operator", "aten.convolution.default"}, {"route", requested}},
      explicit_case.graph,
      explicit_case.args);
  EXPECT_EQ(result.at("requested"), requested);
  EXPECT_EQ(result.at("resolved").at("name"), "im2col");
  EXPECT_EQ(
      result.at("resolved").at("options").at("intermediate_storage"),
      "texture2d");
  EXPECT_EQ(explicit_case.args, original_args);
  EXPECT_EQ(
      explicit_case.graph.sizes_of(original_args[0]),
      (std::vector<int64_t>{1, 4, 7, 7}));
  FpConvCase automatic;
  const Json auto_result = build_operator_route(
      {{"operator", "aten.convolution.default"}},
      automatic.graph,
      automatic.args);
  EXPECT_EQ(auto_result.at("requested").at("name"), "auto");
  EXPECT_TRUE(auto_result.at("requested").at("options").empty());
  EXPECT_EQ(
      auto_result.at("resolved"),
      inspect_conv2d_route(
          automatic.graph,
          automatic.args,
          0,
          automatic.graph.execute_nodes().size()));
  EXPECT_NE(auto_result.at("resolved").at("name"), "unknown");
}

TEST_F(
    BenchmarkOperatorRoute,
    RegistryRejectsUnknownNamesAndAutoOptionsWithoutDispatches) {
  FpConvCase test;
  EXPECT_THROW(
      build_operator_route(
          {{"operator", "aten.convolution.default"},
           {"route",
            {{"name", "not_registered"}, {"options", Json::object()}}}},
          test.graph,
          test.args),
      std::invalid_argument);
  EXPECT_THROW(
      build_operator_route(
          {{"operator", "aten.convolution.default"},
           {"route",
            {{"name", "auto"},
             {"options", {{"intermediate_storage", "buffer"}}}}}},
          test.graph,
          test.args),
      std::invalid_argument);
  EXPECT_TRUE(test.graph.execute_nodes().empty());
  EXPECT_TRUE(test.graph.prepack_nodes().empty());
}

TEST_F(BenchmarkOperatorRoute, GroupedRoutesRejectOutputBlocksCrossingGroups) {
  FpConvCase fp(
      vkapi::kFloat, utils::kTexture3D, utils::kChannelsPacked, 1, 16, 4);
  EXPECT_THROW(
      build_conv2d_direct(fp.graph, fp.args, Json::object()), UnsupportedRoute);
  EXPECT_TRUE(fp.graph.prepack_nodes().empty());
  Q8ConvCase q8(1, 16, 4);
  EXPECT_THROW(
      build_q8ta_conv2d_direct(q8.graph, q8.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_q8ta_conv2d_im2col(q8.graph, q8.args, Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(q8.graph.execute_nodes().empty());
  EXPECT_TRUE(q8.graph.prepack_nodes().empty());
}

TEST_F(
    BenchmarkOperatorRoute,
    ForcedQ8Im2colDoesNotFallBackWhenScratchPlanIsInfeasible) {
  std::vector<int8_t> large_weights(8 * 65536, 1);
  Q8ConvCase test;
  test.args[0] = test.graph.add_tensor(
      {1, 65536, 1, 1024},
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);
  test.args[3] =
      test.graph.add_tensorref({8, 65536}, vkapi::kChar, large_weights.data());
  test.args[9] = test.graph.add_scalar_list(std::vector<int64_t>{1, 1});
  test.args[11] = test.graph.add_scalar_list(std::vector<int64_t>{0, 0});
  test.args[15] = test.graph.add_tensor(
      {1, 8, 1, 1024}, vkapi::kInt8x4, utils::kBuffer, utils::kPackedInt8_4W4C);
  EXPECT_THROW(
      build_q8ta_conv2d_im2col(test.graph, test.args, Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(test.graph.execute_nodes().empty());
  EXPECT_TRUE(test.graph.prepack_nodes().empty());
}

TEST_F(
    BenchmarkOperatorRoute,
    FloatingConstantDtypesMustMatchTheirPrepackShader) {
  FpConvCase mismatched_bias;
  mismatched_bias.args[2] = mismatched_bias.graph.add_tensorref(
      {8}, vkapi::kHalf, mismatched_bias.half_bias.data());
  EXPECT_THROW(
      build_conv2d_direct(
          mismatched_bias.graph, mismatched_bias.args, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      build_conv2d_im2col(
          mismatched_bias.graph, mismatched_bias.args, Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(mismatched_bias.graph.prepack_nodes().empty());
  FpConvCase half_io(vkapi::kHalf);
  if (half_io.graph.dtype_of(half_io.args[0]) == vkapi::kHalf) {
    half_io.args[1] = half_io.graph.add_tensorref(
        {8, 4, 3, 3}, vkapi::kFloat, half_io.weights.data());
    half_io.args[2] =
        half_io.graph.add_tensorref({8}, vkapi::kFloat, half_io.bias.data());
    EXPECT_THROW(
        build_conv2d_direct(half_io.graph, half_io.args, Json::object()),
        UnsupportedRoute);
  }
}

} // namespace
TEST_F(
    BenchmarkOperatorRoute,
    LegacyQdqRejectsUnsupportedFloatingBufferLayout) {
  ComputeGraph graph{GraphConfig{}};
  const auto fp = graph.add_tensor(
      {1, 3, 15, 15}, vkapi::kFloat, utils::kBuffer, utils::kChannelsPacked);
  const auto quant = graph.add_tensor(
      {1, 3, 15, 15}, vkapi::kInt8x4, utils::kBuffer, utils::kPackedInt8_4W4C);
  const auto scale = graph.add_scalar(0.1);
  const auto zero = graph.add_scalar(int64_t(0));
  const auto qmin = graph.add_scalar(int64_t(-128));
  const auto qmax = graph.add_scalar(int64_t(127));
  const auto dtype = graph.add_scalar(int64_t(2));
  const auto out_dtype = graph.add_scalar(int64_t(10));
  const auto quantize = route_registry().operator_route(
      "quantized_decomposed.quantize_per_tensor.default", "legacy_4w4c");
  const auto dequantize = route_registry().operator_route(
      "quantized_decomposed.dequantize_per_tensor.default", "legacy_4w4c");
  EXPECT_THROW(
      quantize(
          graph, {fp, scale, zero, qmin, qmax, dtype, quant}, Json::object()),
      UnsupportedRoute);
  EXPECT_THROW(
      dequantize(
          graph,
          {quant, scale, zero, qmin, qmax, dtype, out_dtype, fp},
          Json::object()),
      UnsupportedRoute);
  EXPECT_TRUE(graph.execute_nodes().empty());
}

} // namespace vkcompute::shader_profiling

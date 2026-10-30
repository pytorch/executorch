// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ReferenceRegistry.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/convolution/conv1d/Conv1dReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/quantization/choose_qparams_per_row/ChooseQparamsReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/binary/add/AddReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/matmul/MatmulReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/embedding/qw/q4gsw/EmbeddingReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswReference.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/operators/attention/sdpa/SdpaReference.h>
#include <gtest/gtest.h>


namespace vkcompute::shader_profiling {
namespace {
Json tensor(const char* name) {
  return {{"tensor", name}};
}
TEST(MigratedReferences, Conv1dDepthwiseBatchStridePadding) {
  const Json spec = {
      {"args",
       {tensor("input"),
        tensor("weight"),
        nullptr,
        {{"int_list", {2}}},
        {{"int_list", {1}}},
        {{"int_list", {1}}},
        {{"bool", false}},
        {{"int_list", {0}}},
        {{"int", 2}},
        tensor("output")}}};
  const reference::TensorMap tensors = {
      {"input",
       {{2, 2, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}},
      {"weight", {{2, 1, 3}, {1, 1, 1, 2, 2, 2}}}};
  const auto out = reference::conv1d_reference(spec, tensors, 2);
  EXPECT_EQ(out.sizes, (std::vector<int64_t>{2, 2, 2}));
  EXPECT_EQ(out.values, (std::vector<double>{3, 9, 22, 42, 19, 33, 54, 90}));
}
TEST(MigratedReferences, BatchedRectangularMatmul) {
  const Json spec = {
      {"operator", "aten.bmm.default"},
      {"args", {tensor("a"), tensor("b"), tensor("output")}}};
  const reference::TensorMap tensors = {
      {"a", {{2, 1, 2}, {1, 2, 3, 4}}},
      {"b", {{2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}}};
  const auto out = reference::matmul_reference(spec, tensors, 2);
  EXPECT_EQ(out.sizes, (std::vector<int64_t>{2, 1, 3}));
  EXPECT_EQ(out.values, (std::vector<double>{9, 12, 15, 61, 68, 75}));
}
TEST(MigratedReferences, AddmmAlphaBeta) {
  const Json spec = {
      {"operator", "aten.addmm.default"},
      {"args",
       {tensor("bias"),
        tensor("a"),
        tensor("b"),
        {{"float", 3}},
        {{"float", 2}},
        tensor("output")}}};
  const reference::TensorMap tensors = {
      {"a", {{1, 2}, {1, 2}}},
      {"b", {{2, 2}, {3, 4, 5, 6}}},
      {"bias", {{2}, {10, 20}}}};
  EXPECT_EQ(
      reference::matmul_reference(spec, tensors, 1).values,
      (std::vector<double>{56, 92}));
}
TEST(MigratedReferences, EmbeddingNibbleOrderAndGroupedScales) {
  Json spec = {
      {"args",
       {tensor("w"),
        tensor("s"),
        {{"int", 2}},
        tensor("i"),
        {{"bool", false}},
        tensor("output")}}};
  const reference::TensorMap tensors = {
      {"w", {{2, 2}, {0x9a, 0xbc, 0xde, 0xf8}}},
      {"s", {{2, 2}, {1, 2, 3, 4}}},
      {"i", {{2}, {1, 0}}}};
  EXPECT_EQ(
      reference::embedding_q4gsw_reference(spec, tensors, 1).values,
      (std::vector<double>{15, 18, 28, 0, 1, 2, 6, 8}));
  spec["args"][4]["bool"] = true;
  EXPECT_EQ(
      reference::embedding_q4gsw_reference(spec, tensors, 1).values,
      (std::vector<double>{18, 15, 0, 28, 2, 1, 8, 6}));
}
TEST(MigratedReferences, DynamicQ4Clipping) {
  Json spec = {
      {"args",
       {tensor("x"),
        tensor("is"),
        tensor("zp"),
        tensor("w"),
        tensor("sum"),
        tensor("s"),
        {{"int", 2}},
        nullptr,
        tensor("output")}},
      {"route", {{"name", "auto"}}}};
  const reference::TensorMap tensors = {
      {"x", {{1, 2}, {100, -100}}},
      {"is", {{1, 1}, {0.5}}},
      {"zp", {{1, 1}, {0}}},
      {"w", {{1, 1}, {0x99}}},
      {"s", {{1, 1}, {1}}}};
  EXPECT_EQ(
      reference::dq8ca_q4gsw_reference(spec, tensors, 1).values,
      (std::vector<double>{-0.5}));
}

TEST(MigratedReferences, DynamicQ4UsesWeightOnlyFallbackWithoutIntegerDot) {
  Json spec = {
      {"args",
       {tensor("x"),
        tensor("is"),
        tensor("zp"),
        tensor("w"),
        tensor("sum"),
        tensor("s"),
        {{"int", 2}},
        nullptr,
        tensor("output")}},
      {"reference_context", {{"int8_dot_product", false}}}};
  const reference::TensorMap tensors = {
      {"x", {{1, 2}, {100, -100}}},
      {"w", {{1, 1}, {0x99}}},
      {"s", {{1, 1}, {1}}}};
  EXPECT_EQ(
      reference::dq8ca_q4gsw_reference(spec, tensors, 1).values,
      (std::vector<double>{0}));
}

TEST(MigratedReferences, DynamicQ4RejectsMalformedReferenceCapabilities) {
  Json spec = {
      {"args",
       {tensor("x"),
        tensor("is"),
        tensor("zp"),
        tensor("w"),
        tensor("sum"),
        tensor("s"),
        {{"int", 2}},
        nullptr,
        tensor("output")}},
      {"reference_context", {{"int8_dot_product", "false"}}}};
  EXPECT_THROW(
      reference::dq8ca_q4gsw_reference(spec, reference::TensorMap{}, 1),
      std::invalid_argument);
}
TEST(MigratedReferences, SdpaFusedCacheUpdateAndCausalPrefix) {
  Json spec = {
      {"operator", "sdpa_with_kv_cache.default"},
      {"args",
       {tensor("q"),
        tensor("pk"),
        tensor("pv"),
        tensor("k"),
        tensor("v"),
        {{"int", 1}},
        {{"int", 2}},
        nullptr,
        {{"float", 0}},
        {{"bool", true}},
        {{"float", 1}},
        tensor("output")}}};
  const reference::TensorMap tensors = {
      {"q", {{1, 2, 2, 1}, {0, 0, 0, 0}}},
      {"k", {{1, 4, 1, 1}, {0, 0, 0, 0}}},
      {"v", {{1, 4, 1, 1}, {2, 100, 200, 1000}}},
      {"pk", {{1, 2, 1, 1}, {0, 0}}},
      {"pv", {{1, 2, 1, 1}, {4, 6}}}};
  const auto out = reference::sdpa_reference(spec, tensors, 2);
  EXPECT_EQ(out.values, (std::vector<double>{3, 3, 4, 4}));
  spec["operator"] = "llama.custom_sdpa.default";
  spec["args"] = {
      tensor("q"),
      tensor("k"),
      tensor("v"),
      {{"int", 1}},
      nullptr,
      {{"float", 0}},
      {{"bool", true}},
      {{"float", 1}},
      tensor("output")};
  const auto unchanged = reference::sdpa_reference(spec, tensors, 1);
  EXPECT_FLOAT_EQ(unchanged.values[0], 51);
  EXPECT_FLOAT_EQ(unchanged.values[2], 302.0f / 3);
}
TEST(MigratedReferences, NativeAddBroadcastAndAlpha) {
  const Json spec = {
      {"operator", "aten.add.Tensor"},
      {"args", {tensor("a"), tensor("b"), {{"float", 0.5}}, tensor("out")}},
      {"tensors", {{"out", {{"sizes", {2, 3}}}}}}};
  const reference::TensorMap tensors = {
      {"a", {{2, 1}, {1, 2}}}, {"b", {{1, 3}, {2, 4, 6}}}};
  const auto out = reference::add_tensor_reference(spec, tensors, 2);
  EXPECT_EQ(out.sizes, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(out.values, (std::vector<double>{2, 3, 4, 3, 4, 5}));
}
TEST(MigratedReferences, ChooseQparamsZeroAndAsymmetricRange) {
  const Json spec = {
      {"args",
       {tensor("x"),
        {{"int", -128}},
        {{"int", 127}},
        tensor("scale"),
        tensor("zero_point")}}};
  const reference::TensorMap tensors = {{"x", {{2, 3}, {0, 0, 0, 0, 1, 2}}}};
  const auto out = reference::choose_qparams_reference(spec, tensors, 2);
  EXPECT_FLOAT_EQ(out.at("scale")[0], 0.1f);
  EXPECT_FLOAT_EQ(out.at("scale")[1], 2.0f / 255);
  EXPECT_EQ(out.at("zero_point"), (std::vector<double>{127, -128}));
}
} // namespace
} // namespace vkcompute::shader_profiling

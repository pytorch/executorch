// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ThroughputRegistry.h>

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

namespace vkcompute::shader_profiling {
namespace {

double memory_count(const Json&) {
  return 40;
}
double compute_count(const Json&) {
  return 20;
}
double negative_count(const Json&) {
  return -1;
}
double infinite_count(const Json&) {
  return std::numeric_limits<double>::infinity();
}
double nan_count(const Json&) {
  return std::numeric_limits<double>::quiet_NaN();
}
double throwing_count(const Json&) {
  throw std::invalid_argument("bad work shape");
}

REGISTER_THROUGHPUT_FUNCTIONS {
  REGISTER_OPERATOR_MEMORY_TRAFFIC(test_counter_case, memory_count);
  REGISTER_OPERATOR_COMPUTE_FLOPS(test_counter_case, compute_count);
  REGISTER_SHADER_MEMORY(test_counter_case, compute_count);
  REGISTER_SHADER_COMPUTE(test_counter_case, memory_count);
  REGISTER_SHADER_MEMORY(test_negative, negative_count);
  REGISTER_SHADER_MEMORY(test_infinite, infinite_count);
  REGISTER_SHADER_MEMORY(test_nan, nan_count);
  REGISTER_SHADER_MEMORY(test_throwing, throwing_count);
  REGISTER_SHADER_COMPUTE(test_throwing, compute_count);
  REGISTER_SHADER_MEMORY_TRAFFIC(test_prefix, memory_count);
  REGISTER_SHADER_COMPUTE_FLOPS(test_prefix, compute_count);
}

Json tensor(
    const Json& sizes,
    const std::string& dtype = "float32",
    const std::string& access = "read",
    const std::string& type = "buffer") {
  return {
      {"sizes", sizes}, {"dtype", dtype}, {"access", access}, {"type", type}};
}

Json linear_case() {
  return {
      {"operator_name", "aten.linear.default"},
      {"args",
       {{{"tensor", tensor({2, 3})}},
        {{"tensor", tensor({4, 3}, "float32", "read", "constant")}},
        {{"tensor", tensor({4}, "float32", "read", "constant")}},
        {{"tensor", tensor({2, 4}, "float32", "write")}}}}};
}

TEST(BenchmarkThroughputRegistry, OperatorAndShaderCountersAreIndependent) {
  ThroughputRegistry registry;
  registry.register_operator_memory("same", memory_count);
  registry.register_operator_compute("same", compute_count);
  registry.register_shader_memory("same", compute_count);
  registry.register_shader_compute("same", memory_count);
  EXPECT_EQ(registry.operator_memory("same"), memory_count);
  EXPECT_EQ(registry.operator_compute("same"), compute_count);
  EXPECT_EQ(registry.shader_memory("same"), compute_count);
  EXPECT_EQ(registry.shader_compute("same"), memory_count);
}

TEST(BenchmarkThroughputRegistry, ShaderPrefixesMatchOnlyStartsOfNames) {
  ThroughputRegistry registry;
  registry.register_shader_memory_prefix("conv2d", memory_count);
  registry.register_shader_compute_prefix("conv2d", compute_count);
  for (const char* name :
       {"conv2d", "conv2d_float", "conv2d_gemm_texture3d_half"}) {
    EXPECT_EQ(registry.shader_memory(name), memory_count);
    EXPECT_EQ(registry.shader_compute(name), compute_count);
  }
  for (const char* name : {"q8ta_conv2d_float", "xconv2d", "Conv2d_float"}) {
    EXPECT_THROW(registry.shader_memory(name), std::invalid_argument);
    EXPECT_THROW(registry.shader_compute(name), std::invalid_argument);
  }
  EXPECT_THROW(registry.operator_memory("conv2d_float"), std::invalid_argument);
}

TEST(
    BenchmarkThroughputRegistry,
    ExactShaderNamesOverrideLongestMatchingPrefix) {
  for (bool broad_first : {false, true}) {
    ThroughputRegistry registry;
    for (const char* prefix : broad_first
             ? std::vector<const char*>{"conv2d", "conv2d_im2col"}
             : std::vector<const char*>{"conv2d_im2col", "conv2d"}) {
      const auto function =
          std::string(prefix) == "conv2d" ? memory_count : compute_count;
      registry.register_shader_memory_prefix(prefix, function);
      registry.register_shader_compute_prefix(prefix, function);
    }
    registry.register_shader_memory(
        "conv2d_im2col_buffer_float", negative_count);
    registry.register_shader_compute(
        "conv2d_im2col_buffer_float", negative_count);
    EXPECT_EQ(registry.shader_memory("conv2d_float"), memory_count);
    EXPECT_EQ(registry.shader_compute("conv2d_float"), memory_count);
    EXPECT_EQ(
        registry.shader_memory("conv2d_im2col_texture3d_half"), compute_count);
    EXPECT_EQ(
        registry.shader_compute("conv2d_im2col_texture3d_half"), compute_count);
    EXPECT_EQ(
        registry.shader_memory("conv2d_im2col_buffer_float"), negative_count);
    EXPECT_EQ(
        registry.shader_compute("conv2d_im2col_buffer_float"), negative_count);
  }
}

TEST(
    BenchmarkThroughputRegistry,
    PrefixRegistrationsRejectInvalidAndDuplicateNames) {
  ThroughputRegistry registry;
  registry.register_shader_memory_prefix("conv2d", memory_count);
  registry.register_shader_compute_prefix("conv2d", memory_count);
  EXPECT_THROW(
      registry.register_shader_memory_prefix("conv2d", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute_prefix("conv2d", compute_count),
      std::invalid_argument);
  EXPECT_EQ(registry.shader_memory("conv2d_half"), memory_count);
  EXPECT_EQ(registry.shader_compute("conv2d_half"), memory_count);
  EXPECT_THROW(
      registry.register_shader_memory_prefix("", memory_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_memory_prefix("conv2d", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute_prefix("", memory_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute_prefix("conv2d", nullptr),
      std::invalid_argument);
}

TEST(BenchmarkThroughputRegistry, NullShaderCountersAreRejectedBeforeLookup) {
  ThroughputRegistry registry;
  registry.register_shader_memory_prefix("conv2d", memory_count);
  registry.register_shader_compute_prefix("conv2d", compute_count);
  EXPECT_THROW(
      registry.register_shader_memory_prefix("conv2d_gemm", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute_prefix("conv2d_gemm", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_memory("conv2d_gemm_half", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute("conv2d_gemm_half", nullptr),
      std::invalid_argument);
  EXPECT_EQ(registry.shader_memory("conv2d_gemm_half"), memory_count);
  EXPECT_EQ(registry.shader_compute("conv2d_gemm_half"), compute_count);
}

TEST(
    BenchmarkThroughputRegistry,
    StaticPrefixCountersReceiveResolvedShaderMetadata) {
  const auto work =
      estimate_work({{"shader_name", "test_prefix_float"}}, true, true);
  EXPECT_EQ(work.memory_bytes, 40);
  EXPECT_EQ(work.compute_flops, 20);
  EXPECT_TRUE(work.diagnostics.empty());
}

TEST(
    BenchmarkThroughputRegistry,
    FamilyPrefixesCoverUnenumeratedShaderVariants) {
  const auto& registry = throughput_registry();
  for (const char* name :
       {"conv2d_gemm_texture2d_half",
        "conv2d_new_variant_float",
        "q8ta_conv2d_float",
        "q8ta_im2col_new_variant",
        "linear_vec_bias_tile_row_4_buffer_texture2d_half",
        "buffer_to_buffer_new_dtype"}) {
    EXPECT_EQ(registry.shader_memory(name), throughput::tensor_traffic);
  }
  EXPECT_EQ(
      registry.shader_compute(
          "linear_vec_bias_tile_row_4_buffer_texture2d_half"),
      registry.shader_compute(
          "linear_vec_bias_tile_row_1_buffer_texture2d_half"));
  EXPECT_EQ(
      registry.shader_compute(
          "linear_vec_tile_row_4_texture3d_texture2d_float"),
      registry.shader_compute(
          "linear_vec_tile_row_1_texture3d_texture2d_float"));
  EXPECT_NE(
      registry.shader_compute("linear_vec_bias_float"),
      registry.shader_compute("linear_vec_float"));
  EXPECT_EQ(
      registry.shader_memory("q4gsw_linear_gemm__w_8x8_nc_buffer_half"),
      registry.shader_memory("q4gsw_linear_gemm__w_4x8_nc_buffer_float"));
  EXPECT_EQ(
      registry.shader_memory("q4gsw_linear_gemm__tin__w_8x8_nc_buffer_half"),
      registry.shader_memory("q4gsw_linear_gemm__tin__w_4x8_nc_buffer_half"));
  EXPECT_NE(
      registry.shader_memory("q4gsw_linear_gemm__tin__float"),
      registry.shader_memory("q4gsw_linear_gemm__float"));
  EXPECT_EQ(
      registry.shader_compute("binary_add_buffer_float"),
      registry.shader_compute("binary_add_texture3d_float"));
}

TEST(
    BenchmarkThroughputRegistry,
    CounterFunctionsCanBeSharedAcrossRegistrations) {
  ThroughputRegistry registry;
  registry.register_operator_memory("first", memory_count);
  registry.register_operator_memory("second", memory_count);
  registry.register_shader_memory("first", memory_count);
  registry.register_shader_compute_prefix("second", memory_count);
  EXPECT_EQ(registry.operator_memory("first"), memory_count);
  EXPECT_EQ(registry.operator_memory("second"), memory_count);
  EXPECT_EQ(registry.shader_memory("first"), memory_count);
  EXPECT_EQ(registry.shader_compute("second_half"), memory_count);
}

TEST(BenchmarkThroughputRegistry, DuplicateRegistrationsKeepOriginalMapping) {
  ThroughputRegistry registry;
  registry.register_operator_memory("same", memory_count);
  registry.register_operator_compute("same", memory_count);
  registry.register_shader_memory("same", memory_count);
  registry.register_shader_compute("same", memory_count);
  EXPECT_THROW(
      registry.register_operator_memory("same", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_compute("same", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_memory("same", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute("same", compute_count),
      std::invalid_argument);
  EXPECT_EQ(registry.operator_memory("same"), memory_count);
  EXPECT_EQ(registry.operator_compute("same"), memory_count);
  EXPECT_EQ(registry.shader_memory("same"), memory_count);
  EXPECT_EQ(registry.shader_compute("same"), memory_count);
}

TEST(BenchmarkThroughputRegistry, InvalidNamesAndFunctionsAreRejected) {
  ThroughputRegistry registry;
  EXPECT_THROW(
      registry.register_operator_memory("", memory_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_compute("", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_memory("", memory_count), std::invalid_argument);
  EXPECT_THROW(
      registry.register_shader_compute("", compute_count),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_memory("same", nullptr),
      std::invalid_argument);
  EXPECT_THROW(
      registry.register_operator_compute("same", nullptr),
      std::invalid_argument);
  EXPECT_THROW(registry.shader_compute("missing"), std::invalid_argument);
  EXPECT_THROW(registry.operator_memory("missing"), std::invalid_argument);
}

TEST(
    BenchmarkThroughputRegistry,
    StaticRegistrationsSelectOperatorAndShaderCounters) {
  const auto op =
      estimate_work({{"operator_name", "test_counter_case"}}, true, true);
  const auto shader =
      estimate_work({{"shader_name", "test_counter_case"}}, true, true);
  EXPECT_EQ(op.memory_bytes, 40);
  EXPECT_EQ(op.compute_flops, 20);
  EXPECT_EQ(shader.memory_bytes, 20);
  EXPECT_EQ(shader.compute_flops, 40);
  EXPECT_TRUE(op.diagnostics.empty());
}

TEST(BenchmarkThroughputRegistry, MissingCountersAreOptionalWithDiagnostics) {
  const auto absent = estimate_work({{"shader_name", "missing"}}, true, true);
  EXPECT_FALSE(absent.memory_bytes);
  EXPECT_FALSE(absent.compute_flops);
  EXPECT_EQ(absent.diagnostics.size(), 2);
  const auto disabled = estimate_work(Json(), false, false);
  EXPECT_FALSE(disabled.memory_bytes);
  EXPECT_TRUE(disabled.diagnostics.empty());
}

TEST(BenchmarkThroughputRegistry, InvalidWorkCountsDoNotHideValidOtherCounter) {
  for (const char* name :
       {"test_negative", "test_infinite", "test_nan", "test_throwing"}) {
    const auto result = estimate_work(
        {{"shader_name", name}}, true, name == std::string("test_throwing"));
    EXPECT_FALSE(result.memory_bytes) << name;
    ASSERT_FALSE(result.diagnostics.empty()) << name;
    if (name == std::string("test_throwing")) {
      EXPECT_EQ(result.compute_flops, 20);
      EXPECT_NE(
          result.diagnostics[0].find("bad work shape"), std::string::npos);
    }
  }
}

TEST(
    BenchmarkThroughputRegistry,
    LinearCountsLogicalTrafficAndMacsIncludingBias) {
  const auto work = estimate_work(linear_case(), true, true);
  EXPECT_EQ(work.memory_bytes, 120);
  EXPECT_EQ(work.compute_flops, 56);
  EXPECT_TRUE(work.diagnostics.empty());
  Json half = linear_case();
  for (auto& arg : half["args"]) {
    arg["tensor"]["dtype"] = "float16";
  }
  EXPECT_EQ(estimate_work(half, true, false).memory_bytes, 60);
  Json no_bias = linear_case();
  no_bias["args"][2] = nullptr;
  const auto without_bias = estimate_work(no_bias, true, true);
  EXPECT_EQ(without_bias.memory_bytes, 104);
  EXPECT_EQ(without_bias.compute_flops, 48);
}

TEST(
    BenchmarkThroughputRegistry,
    Q4CountsPackedBytesAndArithmeticEquivalentMacs) {
  const Json spec = {
      {"operator_name", "et_vk.q4gsw_linear.default"},
      {"args",
       {{{"tensor", tensor({2, 32})}},
        {{"tensor", tensor({4, 16}, "uint8", "read", "constant")}},
        {{"tensor", tensor({1, 4}, "float32", "read", "constant")}},
        {{"int", 32}},
        {{"tensor", tensor({4}, "float32", "read", "constant")}},
        {{"tensor", tensor({2, 4}, "float32", "write")}}}}};
  const auto work = estimate_work(spec, true, true);
  EXPECT_EQ(work.memory_bytes, 384);
  EXPECT_EQ(work.compute_flops, 520);
}

TEST(
    BenchmarkThroughputRegistry,
    ConvolutionCountsGroupedStrideAndPaddingCase) {
  const Json spec = {
      {"operator_name", "aten.convolution.default"},
      {"args",
       {{{"tensor", tensor({1, 4, 5, 5})}},
        {{"tensor", tensor({8, 2, 3, 3}, "float32", "read", "constant")}},
        {{"tensor", tensor({8}, "float32", "read", "constant")}},
        {{"int_list", {2, 2}}},
        {{"int_list", {1, 1}}},
        {{"int_list", {1, 1}}},
        {{"bool", false}},
        {{"int_list", {0, 0}}},
        {{"int", 2}},
        {{"tensor", tensor({1, 8, 3, 3}, "float32", "write")}}}}};
  const auto work = estimate_work(spec, true, true);
  EXPECT_EQ(work.memory_bytes, 1296);
  EXPECT_EQ(work.compute_flops, 2664);
  Json transposed = spec;
  transposed["args"][6]["bool"] = true;
  EXPECT_FALSE(estimate_work(transposed, false, true).compute_flops);
}

TEST(
    BenchmarkThroughputRegistry,
    Q8CountsPackedInt8ElementsAndCorrectionScales) {
  const Json spec = {
      {"operator_name", "et_vk.q8ta_conv2d.default"},
      {"args",
       {{{"tensor", tensor({1, 4, 5, 5}, "int8x4")}},
        {{"float", 0.25}},
        {{"int", 0}},
        {{"tensor", tensor({8, 36}, "int8", "read", "constant")}},
        {{"tensor", tensor({8}, "int32", "read", "constant")}},
        {{"tensor", tensor({8}, "float32", "read", "constant")}},
        {{"float", 0.25}},
        {{"int", 0}},
        {{"tensor", tensor({8}, "float32", "read", "constant")}},
        {{"int_list", {3, 3}}},
        {{"int_list", {1, 1}}},
        {{"int_list", {1, 1}}},
        {{"int_list", {1, 1}}},
        {{"int", 1}},
        {{"string", "none"}},
        {{"tensor", tensor({1, 8, 5, 5}, "int8x4", "write")}}}}};
  const auto work = estimate_work(spec, true, true);
  EXPECT_EQ(work.memory_bytes, 684);
  EXPECT_EQ(work.compute_flops, 14600);
}

TEST(
    BenchmarkThroughputRegistry,
    CopyAndBroadcastAdditionHaveLogicalWorkCounts) {
  const Json copy = {
      {"shader_name", "buffer_to_buffer_float"},
      {"bindings",
       {{"input", tensor({8})},
        {"output", tensor({8}, "float32", "write")},
        {"meta", {{"type", "meta_ubo"}}}}}};
  const auto copied = estimate_work(copy, true, true);
  EXPECT_EQ(copied.memory_bytes, 64);
  EXPECT_EQ(copied.compute_flops, 0);
  const Json add = {
      {"shader_name", "binary_add_texture3d_float"},
      {"bindings",
       {{"a", tensor({2, 1}, "float32", "read", "texture3d")},
        {"b", tensor({3}, "float32", "read", "texture3d")},
        {"out", tensor({2, 3}, "float32", "write", "texture3d")}}}};
  const auto added = estimate_work(add, true, true);
  EXPECT_EQ(added.memory_bytes, 44);
  EXPECT_EQ(added.compute_flops, 12);
}

TEST(
    BenchmarkThroughputRegistry,
    ShaderLinearCountsActualBindingsWithoutParentWork) {
  const Json spec = {
      {"shader_name", "linear_vec_bias_tile_row_1_buffer_texture2d_float"},
      {"bindings",
       {{"binding_0", tensor({1, 8}, "float32", "write")},
        {"binding_1", tensor({1, 4})},
        {"binding_2", tensor({8, 4}, "float32", "read", "texture2d")},
        {"binding_3", tensor({8})}}}};
  const auto work = estimate_work(spec, true, true);
  EXPECT_EQ(work.memory_bytes, 208);
  EXPECT_EQ(work.compute_flops, 72);
  EXPECT_TRUE(work.diagnostics.empty());
  Json missing = spec;
  missing["shader_name"] = "unregistered_child_shader";
  EXPECT_FALSE(estimate_work(missing, true, true).memory_bytes);
}

TEST(
    BenchmarkThroughputRegistry,
    LogicalTrafficCountsReadWriteTwiceAndDeduplicatesRefs) {
  const Json spec = {
      {"shader_name", "buffer_to_buffer_float"},
      {"bindings",
       {{"x",
         {{"name", "x"},
          {"type", "buffer"},
          {"sizes", {8}},
          {"dtype", "float32"},
          {"access", "read_write"}}},
        {"x_alias",
         {{"ref", "x"},
          {"type", "buffer"},
          {"sizes", {8}},
          {"dtype", "float32"},
          {"access", "read"}}}}}};
  EXPECT_EQ(estimate_work(spec, true, false).memory_bytes, 64);
  Json malformed = spec;
  malformed["bindings"]["x"]["dtype"] = "unknown";
  const auto result = estimate_work(malformed, true, false);
  EXPECT_FALSE(result.memory_bytes);
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(BenchmarkThroughputRegistry, Q4ShaderMemoryExcludesInactiveBindingSlots) {
  Json fp32 = {
      {"shader_name", "q4gsw_linear_gemm__w_4x8_nc_buffer_float"},
      {"bindings",
       {{"binding_0", tensor({4, 8}, "float32", "write")},
        {"binding_1", tensor({4, 32})},
        {"binding_2", tensor(Json::array())},
        {"binding_3", tensor({8, 4}, "int32")},
        {"binding_4", tensor({1, 8})},
        {"binding_5", tensor({8})}}}};
  for (size_t i = 0; i < 6; ++i) {
    fp32["bindings"]["binding_" + std::to_string(i)]["binding_index"] = i;
  }
  const auto fp32_work = estimate_work(fp32, true, true);
  EXPECT_EQ(fp32_work.memory_bytes, 832);
  EXPECT_FALSE(fp32_work.compute_flops);
  Json fp16 = fp32;
  fp16["shader_name"] = "q4gsw_linear_gemm__tin__w_4x8_nc_buffer_half";
  fp16["bindings"]["binding_2"]["sizes"] = {32, 4};
  for (size_t i : {0, 1, 2, 4, 5}) {
    fp16["bindings"]["binding_" + std::to_string(i)]["dtype"] = "float16";
  }
  const auto fp16_work = estimate_work(fp16, true, false);
  EXPECT_EQ(fp16_work.memory_bytes, 480);
  EXPECT_TRUE(fp16_work.diagnostics.empty());
  fp16["bindings"]["binding_5"]["sizes"] = Json::array();
  EXPECT_EQ(estimate_work(fp16, true, false).memory_bytes, 464);
}

} // namespace
} // namespace vkcompute::shader_profiling

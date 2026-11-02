// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/test/api_abstractions/VulkanTest.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/OperatorBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/ShaderBenchmark.h>
#include <executorch/backends/vulkan/test/ops/benchmark_v2/framework/Validation.h>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace vkcompute::shader_profiling {
using BenchmarkValidation = VulkanTest;

namespace {

Json copy_case(const Json& gwg) {
  return parse_configs({{"shader", "buffer_to_buffer_float"},
                        {"gwg_size", gwg},
                        {"lwg_size", {1, 1, 1}},
                        {"bindings",
                         Json::array(
                             {{{"buffer",
                                {{"name", "out"},
                                 {"sizes", {8}},
                                 {"access", "write"},
                                 {"data", {{"fill", 0}}}}}},
                              {{"buffer",
                                {{"name", "in"},
                                 {"sizes", {8}},
                                 {"data", {1, 2, 3, 4, 5, 6, 7, 8}}}}},
                              {{"meta_ubo", "out"}, {"field", "numel"}}})}})
      .at(0);
}

TEST_F(BenchmarkValidation, StagedShaderCopyPassesWithoutBenchmarking) {
  const Json spec = copy_case({8, 1, 1});
  const Report result = benchmark_shader(spec, prepare_tensor_data(spec), true);
  EXPECT_EQ(result.at("validation").at("status"), "PASS");
  EXPECT_FALSE(result.contains("gpu_timestamp_shader_us"));
  EXPECT_FALSE(result.contains("wall_time_shader_us"));
}

TEST_F(BenchmarkValidation, GraphCreationReusesPreparedHostData) {
  const Json spec = copy_case({8, 1, 1});
  auto host_data = prepare_tensor_data(spec);
  host_data.at("in") =
      make_data({{"data", {-1, -2, -3, -4, -5, -6, -7, -8}}}, 8);
  const auto expected = host_data;
  const auto* input_data = host_data.at("in").data();
  {
    ComputeGraph graph{GraphConfig{}};
    create_tensors(spec, graph, host_data, true);
    EXPECT_EQ(host_data, expected);
    EXPECT_EQ(host_data.at("in").data(), input_data);
  }
  {
    ComputeGraph graph{GraphConfig{}};
    create_tensors(spec, graph, host_data, false);
    EXPECT_EQ(host_data, expected);
    EXPECT_EQ(host_data.at("in").data(), input_data);
    EXPECT_TRUE(graph.inputs().empty());
    EXPECT_TRUE(graph.execute_nodes().empty());
  }
}

TEST_F(BenchmarkValidation, ValidationSelectionRequiresWritableTensors) {
  Json spec = copy_case({8, 1, 1});
  spec["validation_method"] = {{"function", "is_close"}, {"outputs", {"out"}}};
  ComputeGraph graph{GraphConfig{}};
  const auto host_data = prepare_tensor_data(spec);
  const auto tensors = create_tensors(spec, graph, host_data);
  const Json accesses{{"out", "write"}, {"in", "read"}};
  const auto outputs = add_validation_outputs(spec, graph, tensors, accesses);
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs.begin()->first, "out");
  spec["validation_method"]["outputs"] = {"in"};
  EXPECT_THROW(
      add_validation_outputs(spec, graph, tensors, accesses),
      std::invalid_argument);
}

TEST_F(BenchmarkValidation, RejectsWrongOutputShapeWithMatchingElementCount) {
  Json spec = copy_case({8, 1, 1});
  spec["tensors"]["in"]["sizes"] = {1, 8};
  spec["tensors"]["out"]["sizes"] = {1, 8};
  ComputeGraph graph{GraphConfig{}};
  const auto host_data = prepare_tensor_data(spec);
  const auto tensors = create_tensors(spec, graph, host_data);
  graph.virtual_resize(tensors.at("out"), {2, 4});
  const ValueRef staging =
      graph.add_staging(vkapi::kFloat, 8, vkapi::CopyDirection::DEVICE_TO_HOST);
  graph.prepare();
  const std::array<float, 8> values = {1, 2, 3, 4, 5, 6, 7, 8};
  graph.maybe_cast_and_copy_into_staging(
      staging, values.data(), values.size(), vkapi::kFloat);
  EXPECT_THROW(
      validate_outputs(spec, graph, tensors, {{"out", staging}}, host_data),
      std::invalid_argument);
}

TEST_F(BenchmarkValidation, IncompleteShaderGridReportsOutputMismatches) {
  const Json spec = copy_case({1, 1, 1});
  const Report result = benchmark_shader(spec, prepare_tensor_data(spec), true);
  EXPECT_EQ(result.at("validation").at("status"), "FAIL");
  EXPECT_EQ(
      result.at("validation").at("outputs").at("out").at("mismatch_count"), 7);
}

TEST_F(BenchmarkValidation, StagedTextureAdditionUsesOneTargetDescriptorSet) {
  const Json spec =
      parse_configs(
          {{"shader", "binary_add_texture3d_float"},
           {"gwg_size", "default_pick_gwg"},
           {"lwg_size", {10, 10, 1}},
           {"bindings",
            Json::array(
                {{{"texture3d",
                   {{"name", "out"},
                    {"sizes", {1, 1, 1, 8}},
                    {"access", "write"},
                    {"data", {{"fill", 0}}}}}},
                 {{"texture3d",
                   {{"name", "a"},
                    {"sizes", {1, 1, 1, 8}},
                    {"data", {1, 2, 3, 4, 5, 6, 7, 8}}}}},
                 {{"texture3d",
                   {{"name", "b"},
                    {"sizes", {1, 1, 1, 8}},
                    {"data", {-1, -2, -3, -4, -5, -6, -7, -8}}}}},
                 {{"meta_ubo", "out"}},
                 {{"meta_ubo", "a"}},
                 {{"meta_ubo", "b"}}})},
           {"specialization_constants",
            {{{"layout", "out"}}, {{"layout", "a"}}, {{"layout", "b"}}}},
           {"push_constants", {{{"dtype", "float32"}, {"data", {2}}}}}})
          .at(0);
  EXPECT_EQ(
      benchmark_shader(spec, prepare_tensor_data(spec), true)
          .at("validation")
          .at("status"),
      "PASS");
}

TEST_F(BenchmarkValidation, StagedLinearUsesConstantWeightsAndOptionalBias) {
  const Json spec =
      parse_configs(
          {{"operator", "aten.linear.default"},
           {"args_sweep",
            Json::array({Json::array(
                {{{"tensor",
                   {{"name", "x"}, {"sizes", {1, 4}}, {"data", {1, 2, 3, 4}}}}},
                 {{"tensor",
                   {{"name", "w"},
                    {"sizes", {4, 4}},
                    {"storage", "constant"},
                    {"data",
                     {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}}}}},
                 nullptr,
                 {{"tensor",
                   {{"name", "y"},
                    {"sizes", {1, 4}},
                    {"data", {{"fill", 0}}}}}}})})}})
          .at(0);
  for (const char* dtype : {"float32", "float16"}) {
    Json typed = spec;
    for (auto& tensor : typed["tensors"].items()) {
      tensor.value()["dtype"] = dtype;
    }
    typed.update(
        {{"target_ms", 0.01},
         {"benchmark_runs", 2},
         {"warmup_runs", 0},
         {"max_dispatches", 2}});
    const auto host_data = prepare_tensor_data(typed);
    const auto expected = host_data;
    EXPECT_EQ(
        benchmark_operator(typed, host_data, true)
            .at("validation")
            .at("status"),
        "PASS");
    EXPECT_TRUE(
        benchmark_operator(typed, host_data).contains("wall_time_operator_us"));
    EXPECT_EQ(
        benchmark_operator(typed, host_data, true)
            .at("validation")
            .at("status"),
        "PASS");
    EXPECT_EQ(host_data, expected);
    EXPECT_EQ(
        host_data.at("w").size(), std::string(dtype) == "float16" ? 32 : 64);
  }
}

TEST_F(BenchmarkValidation, BenchmarkAddsRegisteredWorkAndWallThroughput) {
  Json spec = copy_case({8, 1, 1});
  spec.update(
      {{"show_memory_throughput", true},
       {"show_compute_throughput", true},
       {"target_ms", 0.01},
       {"benchmark_runs", 2},
       {"warmup_runs", 0},
       {"max_dispatches", 2}});
  const Report result = benchmark_shader(spec, prepare_tensor_data(spec));
  const auto& throughput = result.at("throughput");
  EXPECT_DOUBLE_EQ(throughput.at("memory_bytes"), 64);
  EXPECT_DOUBLE_EQ(throughput.at("compute_flops"), 0);
  EXPECT_EQ(throughput.at("wall_time").at("memory_gib_per_s").size(), 2);
  EXPECT_EQ(
      throughput.at("wall_time").at("compute_gflops"), Report::array({0, 0}));
  const double count = result.at("dispatches_per_batch");
  for (size_t i = 0; i < 2; ++i) {
    const double expected = 64.0 * count * 1e6 /
        result.at("wall_time_batch_us")[i].get<double>() / 1073741824.0;
    EXPECT_NEAR(
        throughput.at("wall_time").at("memory_gib_per_s")[i].get<double>(),
        expected,
        1e-12);
  }
}

TEST_F(BenchmarkValidation, UnsupportedShaderIsRejectedBeforePipelineCreation) {
  ComputeGraph graph{GraphConfig{}};
  if (graph.context()->adapter_ptr()->supports_int8_dot_product()) {
    GTEST_SKIP() << "Requires a device without integer dot-product support";
  }
  const auto shader = VK_KERNEL_FROM_STR("q8ta_conv2d_pw_float");
  ASSERT_TRUE(shader.requires_integer_dot_product);
  EXPECT_THROW(
      graph.register_pipeline_to_create(
          shader, LocalWorkGroup(1, 1, 1), {}, {}),
      vkapi::ShaderNotSupportedError);
}

TEST_F(
    BenchmarkValidation,
    UnrecognizedSpirvCapabilityIsRejectedBeforePipelineCreation) {
  ComputeGraph graph{GraphConfig{}};
  vkapi::ShaderInfo shader;
  shader.kernel_name = "unsupported_capability";
  shader.required_capabilities = {1, 65535};
  EXPECT_THROW(
      graph.register_pipeline_to_create(
          shader, LocalWorkGroup(1, 1, 1), {}, {}),
      vkapi::ShaderNotSupportedError);
}

} // namespace
} // namespace vkcompute::shader_profiling

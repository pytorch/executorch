/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/serialization/schema_generated.h>
#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/relu/relu_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sigmoid/sigmoid_wgsl.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {
namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
WGPUDevice g_device = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

struct UnaryParams {
  uint32_t num_elements;
  float min;
  float max;
  uint32_t padding;
};

WGPUBuffer create_storage_buffer(size_t nbytes) {
  WGPUBufferDescriptor descriptor = {};
  descriptor.size = nbytes;
  descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
  WGPUBuffer buffer = wgpuDeviceCreateBuffer(g_device, &descriptor);
  if (buffer == nullptr) {
    throw std::runtime_error("failed to create compute-dispatch test buffer");
  }
  return buffer;
}

enum class Q4RouteSignal { Static, LegacyGraphMarker, ExplicitOption };

void build_q4_route_graph(WebGPUGraph& graph, Q4RouteSignal signal) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  auto add_tensor = [&](vk::VkDataType dtype,
                        const std::vector<uint32_t>& dims,
                        int mem_obj_id) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb, dtype, &dims, /*constant_id=*/-1, mem_obj_id)
            .Union()));
    return id;
  };
  auto add_int = [&](int64_t value) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, value).Union()));
    return id;
  };

  const int input = add_tensor(vk::VkDataType::FLOAT32, {2, 8}, 0);
  const int weight = add_tensor(vk::VkDataType::UINT8, {8, 4}, 1);
  const int scales = add_tensor(vk::VkDataType::FLOAT32, {1, 8}, 2);
  const int group_size = add_int(8);
  const int bias = static_cast<int>(values.size());
  values.push_back(vk::CreateVkValue(fbb));
  const int output = add_tensor(vk::VkDataType::FLOAT32, {2, 8}, 3);

  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  int q4_input = input;
  if (signal == Q4RouteSignal::LegacyGraphMarker) {
    const int dim = add_int(0);
    const int symint = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::SymInt, vk::CreateSymInt(fbb, 0).Union()));
    const int alpha = add_int(1);
    const int intermediate = add_tensor(vk::VkDataType::FLOAT32, {2, 8}, 4);
    const std::vector<int32_t> sym_size_args = {input, dim, symint};
    chain.push_back(
        vk::CreateOperatorCallDirect(fbb, 0, "sym_size.int", &sym_size_args));
    const std::vector<int32_t> add_args = {input, input, alpha, intermediate};
    chain.push_back(
        vk::CreateOperatorCallDirect(fbb, 1, "aten.add.Tensor", &add_args));
    q4_input = intermediate;
  }

  const std::vector<int32_t> q4_args = {
      q4_input, weight, scales, group_size, bias, output};
  chain.push_back(vk::CreateOperatorCallDirect(
      fbb,
      static_cast<uint32_t>(chain.size()),
      "et_vk.linear_q4gsw.default",
      &q4_args));
  const std::vector<uint32_t> input_ids = {
      static_cast<uint32_t>(input),
      static_cast<uint32_t>(weight),
      static_cast<uint32_t>(scales)};
  const std::vector<uint32_t> output_ids = {static_cast<uint32_t>(output)};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraphConfig config;
  if (signal == Q4RouteSignal::ExplicitOption) {
    config.record_q4gsw_decode_route = true;
  }
  graph.build(fbb.GetBufferPointer(), nullptr, nullptr, config);
}

std::vector<const WebGPUDispatch*> q4_dispatches(WebGPUGraph& graph) {
  std::vector<const WebGPUDispatch*> dispatches;
  for (size_t i = 0; i < graph.num_dispatches(); i++) {
    const WebGPUDispatch& dispatch = graph.dispatch_at(i);
    if (dispatch.kernel_name.rfind("linear_q4gsw", 0) == 0) {
      dispatches.push_back(&dispatch);
    }
  }
  return dispatches;
}

void expect_dual_q4_topology(WebGPUGraph& graph) {
  const auto dispatches = q4_dispatches(graph);
  ASSERT_EQ(dispatches.size(), 2);
  EXPECT_NE(dispatches[0]->pipeline, nullptr);
  EXPECT_NE(dispatches[1]->pipeline, nullptr);
  EXPECT_NE(dispatches[0]->pipeline, dispatches[1]->pipeline);
  EXPECT_NE(dispatches[0]->bind_group, nullptr);
  EXPECT_EQ(dispatches[0]->bind_group, dispatches[1]->bind_group);
  EXPECT_EQ(
      std::count_if(
          dispatches.begin(),
          dispatches.end(),
          [](const WebGPUDispatch* dispatch) {
            return dispatch->workgroup_count_x != 0 &&
                dispatch->workgroup_count_y != 0;
          }),
      1);
}

TEST(WebGPUShaderRegistry, FindsKnownShaderAndRejectsUnknownName) {
  const WebGPUShaderInfo& sigmoid = get_webgpu_shader_info("sigmoid");
  EXPECT_EQ(sigmoid.name, "sigmoid");
  EXPECT_NE(sigmoid.source, nullptr);
  EXPECT_GT(sigmoid.workgroup_size_x, 0u);
  EXPECT_THROW(
      get_webgpu_shader_info("not_a_registered_shader"), std::runtime_error);
}

TEST(WebGPUQ4RouteSignal, PreservesStaticAndRecordsBothDynamicSignals) {
  WebGPUGraph static_graph;
  static_graph.set_device(g_device);
  build_q4_route_graph(static_graph, Q4RouteSignal::Static);
  const auto static_dispatches = q4_dispatches(static_graph);
  ASSERT_EQ(static_dispatches.size(), 1);
  EXPECT_EQ(static_graph.num_dispatches(), 1);
  EXPECT_NE(static_dispatches[0]->pipeline, nullptr);
  EXPECT_NE(static_dispatches[0]->bind_group, nullptr);
  EXPECT_FALSE(static_graph.has_dynamic_shapes());
  EXPECT_FALSE(static_graph.config().record_q4gsw_decode_route);

  WebGPUGraph legacy_graph;
  legacy_graph.set_device(g_device);
  build_q4_route_graph(legacy_graph, Q4RouteSignal::LegacyGraphMarker);
  EXPECT_TRUE(legacy_graph.has_dynamic_shapes());
  EXPECT_FALSE(legacy_graph.config().record_q4gsw_decode_route);
  EXPECT_EQ(legacy_graph.num_dispatches(), 3);
  expect_dual_q4_topology(legacy_graph);

  WebGPUGraph explicit_graph;
  explicit_graph.set_device(g_device);
  build_q4_route_graph(explicit_graph, Q4RouteSignal::ExplicitOption);
  EXPECT_FALSE(explicit_graph.has_dynamic_shapes());
  EXPECT_TRUE(explicit_graph.config().record_q4gsw_decode_route);
  EXPECT_EQ(explicit_graph.num_dispatches(), 2);
  expect_dual_q4_topology(explicit_graph);
}

TEST(WebGPUComputeDispatch, PipelineKeyCanonicalizesConstants) {
  WebGPUComputeDispatchDescriptor first;
  first.shader_name = "sigmoid";
  first.entry_point = "main";
  first.constants = {{"beta", 2.0}, {"alpha", 1.0}};

  WebGPUComputeDispatchDescriptor reordered = first;
  reordered.constants = {{"alpha", 1.0}, {"beta", 2.0}};

  EXPECT_EQ(
      make_compute_pipeline_key(first), make_compute_pipeline_key(reordered));
}

TEST(WebGPUComputeDispatch, PipelineKeyTracksCompileIdentityOnly) {
  WebGPUComputeDispatchDescriptor base;
  base.shader_name = "sigmoid";
  base.entry_point = "main";
  base.constants = {{"wg_size", 64.0}};
  base.grid = {1u, 1u};
  base.bindings = {{reinterpret_cast<WGPUBuffer>(1), 0u, 16u}};

  WebGPUComputeDispatchDescriptor runtime_change = base;
  runtime_change.grid = {17u, 3u};
  runtime_change.bindings = {{reinterpret_cast<WGPUBuffer>(2), 128u, 4096u}};
  EXPECT_EQ(
      make_compute_pipeline_key(base),
      make_compute_pipeline_key(runtime_change));

  WebGPUComputeDispatchDescriptor shader_change = base;
  shader_change.shader_name = "binary_add";
  EXPECT_NE(
      make_compute_pipeline_key(base),
      make_compute_pipeline_key(shader_change));

  WebGPUComputeDispatchDescriptor entry_change = base;
  entry_change.entry_point = "alternate";
  EXPECT_NE(
      make_compute_pipeline_key(base), make_compute_pipeline_key(entry_change));

  WebGPUComputeDispatchDescriptor constant_change = base;
  constant_change.constants = {{"wg_size", 128.0}};
  EXPECT_NE(
      make_compute_pipeline_key(base),
      make_compute_pipeline_key(constant_change));
}

TEST(WebGPUComputeDispatch, PipelineKeyRejectsInvalidConstants) {
  WebGPUComputeDispatchDescriptor duplicate;
  duplicate.shader_name = "sigmoid";
  duplicate.constants = {{"wg_size", 64.0}, {"wg_size", 128.0}};
  EXPECT_THROW(make_compute_pipeline_key(duplicate), std::runtime_error);

  WebGPUComputeDispatchDescriptor non_finite;
  non_finite.shader_name = "sigmoid";
  non_finite.constants = {{"wg_size", std::numeric_limits<double>::infinity()}};
  EXPECT_THROW(make_compute_pipeline_key(non_finite), std::runtime_error);
}

TEST(WebGPUComputeDispatch, DescriptorRejectsInvalidBindings) {
  WebGPUComputeDispatchDescriptor null_buffer;
  null_buffer.shader_name = "sigmoid";
  null_buffer.bindings = {{nullptr, 0u, 16u}};
  EXPECT_THROW(
      validate_compute_dispatch_descriptor(null_buffer), std::runtime_error);

  WebGPUComputeDispatchDescriptor zero_size;
  zero_size.shader_name = "sigmoid";
  zero_size.bindings = {{reinterpret_cast<WGPUBuffer>(1), 0u, 0u}};
  EXPECT_THROW(
      validate_compute_dispatch_descriptor(zero_size), std::runtime_error);

  WebGPUComputeDispatchDescriptor overflow;
  overflow.shader_name = "sigmoid";
  overflow.bindings = {
      {reinterpret_cast<WGPUBuffer>(1),
       std::numeric_limits<uint64_t>::max(),
       2u}};
  EXPECT_THROW(
      validate_compute_dispatch_descriptor(overflow), std::runtime_error);
}

TEST(WebGPUComputeDispatch, ReusesPipelineAndReleasesDawnObjects) {
  constexpr size_t kNumElements = 64;
  constexpr size_t kBufferBytes = kNumElements * sizeof(float);

  for (int iteration = 0; iteration < 32; iteration++) {
    WGPUBuffer input = create_storage_buffer(kBufferBytes);
    WGPUBuffer output = create_storage_buffer(kBufferBytes);
    {
      WebGPUGraph graph;
      graph.set_device(g_device);
      WGPUBuffer clamp_params =
          graph.create_params_buffer(UnaryParams{kNumElements, -1.0f, 1.0f, 0});
      WGPUBuffer hardtanh_params =
          graph.create_params_buffer(UnaryParams{kNumElements, -2.0f, 2.0f, 0});

      WebGPUComputeDispatchDescriptor descriptor;
      descriptor.shader_name = "clamp";
      descriptor.kernel_name = "clamp_test";
      descriptor.bindings = {
          {input, 0u, kBufferBytes},
          {output, 0u, kBufferBytes},
          {clamp_params, 0u, sizeof(UnaryParams)}};
      descriptor.constants = {{"wg_size", 64.0}};
      descriptor.grid = {7u, 3u};

      const size_t first = graph.add_compute_dispatch(descriptor);
      descriptor.kernel_name = "hardtanh_test";
      descriptor.bindings[2].buffer = hardtanh_params;
      const size_t second = graph.add_compute_dispatch(descriptor);

      const WebGPUMemoryStats stats = graph.memory_stats();
      EXPECT_EQ(stats.num_dispatches, 2);
      EXPECT_EQ(stats.num_cached_shaders, 1);
      EXPECT_EQ(stats.num_cached_pipelines, 1);
      EXPECT_EQ(stats.uniform_buffer_bytes, 2 * sizeof(UnaryParams));
      EXPECT_EQ(
          graph.dispatch_at(first).pipeline,
          graph.dispatch_at(second).pipeline);
      EXPECT_EQ(graph.dispatch_at(first).kernel_name, "clamp_test");
      EXPECT_EQ(graph.dispatch_at(second).kernel_name, "hardtanh_test");
      EXPECT_EQ(graph.dispatch_at(first).workgroup_count_x, 7u);
      EXPECT_EQ(graph.dispatch_at(first).workgroup_count_y, 3u);
      EXPECT_EQ(graph.dispatch_at(second).workgroup_count_x, 7u);
      EXPECT_EQ(graph.dispatch_at(second).workgroup_count_y, 3u);
    }
    wgpuBufferRelease(output);
    wgpuBufferRelease(input);
  }
}

TEST(WebGPUComputeDispatch, AlternateShaderReusesLayoutAndBindGroup) {
  constexpr size_t kNumElements = 64;
  constexpr size_t kBufferBytes = kNumElements * sizeof(float);
  WGPUBuffer input = create_storage_buffer(kBufferBytes);
  WGPUBuffer output = create_storage_buffer(kBufferBytes);
  {
    WebGPUGraph graph;
    graph.set_device(g_device);
    WGPUBuffer params =
        graph.create_params_buffer(UnaryParams{kNumElements, 0.0f, 0.0f, 0});
    const std::vector<utils::BindingSpec> bindings = {
        {0, WGPUBufferBindingType_ReadOnlyStorage, input, kBufferBytes},
        {1, WGPUBufferBindingType_Storage, output, kBufferBytes},
        {2, WGPUBufferBindingType_Uniform, params, sizeof(UnaryParams)},
    };
    const WGPUConstantEntry wg_size = utils::make_wg_size_constant(64u);
    {
      utils::ComputePipelineBundle sigmoid = utils::make_compute_pipeline(
          g_device, kSigmoidWGSL, bindings, &wg_size, 1u);
      utils::ComputePipelineBundle relu = utils::make_compute_pipeline(
          g_device, kReluWGSL, sigmoid, &wg_size, 1u);

      EXPECT_EQ(relu.bind_group, sigmoid.bind_group);
      EXPECT_EQ(relu.bind_group_layout, nullptr);
      EXPECT_EQ(relu.pipeline_layout, nullptr);
      EXPECT_NE(relu.shader, nullptr);
      EXPECT_NE(relu.pipeline, nullptr);
      EXPECT_NE(relu.pipeline, sigmoid.pipeline);

      graph.add_dispatch(
          {sigmoid.pipeline, sigmoid.bind_group, 1u, "sigmoid_test"});
      graph.add_dispatch({relu.pipeline, relu.bind_group, 1u, "relu_test"});
    }
  }
  wgpuBufferRelease(output);
  wgpuBufferRelease(input);
}

TEST(WebGPUComputeDispatch, RejectsBindingRangeBeyondDawnBuffer) {
  WGPUBuffer buffer = create_storage_buffer(16u);
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = "sigmoid";
  descriptor.bindings = {{buffer, 8u, 12u}};

  EXPECT_THROW(
      validate_compute_dispatch_descriptor(descriptor), std::runtime_error);
  wgpuBufferRelease(buffer);
}

TEST(WebGPUExecution, FullySuppressedPlanPerformsNoQueueSubmission) {
  WebGPUGraph graph;
  const WebGPUExecutionPlan plan;

  EXPECT_EQ(graph.execute(plan), 0u);
}

TEST(WebGPUExecution, RejectsPlanOutputCountMismatch) {
  WebGPUGraph graph;
  WebGPUExecutionPlan plan;
  plan.copy_outputs = {true};
  std::vector<std::pair<void*, size_t>> outputs;

  EXPECT_THROW(graph.execute(plan), std::runtime_error);
  EXPECT_THROW(graph.copy_outputs(outputs, plan), std::runtime_error);
}

TEST(WebGPUExecution, RejectsPlanDispatchOutOfRange) {
  WebGPUGraph graph;
  WebGPUExecutionPlan plan;
  plan.dispatch_chunks = {{0u}};

  EXPECT_THROW(graph.execute(plan), std::runtime_error);
}

} // namespace
} // namespace executorch::backends::webgpu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  executorch::backends::webgpu::WebGPUContext context;
  try {
    context = executorch::backends::webgpu::create_webgpu_context();
  } catch (const std::exception& error) {
    std::printf("SKIP: %s\n", error.what());
    return 0;
  }
  executorch::backends::webgpu::set_default_webgpu_context(&context);
  executorch::backends::webgpu::g_device = context.device;

  const int result = RUN_ALL_TESTS();
  executorch::backends::webgpu::set_default_webgpu_context(nullptr);
  executorch::backends::webgpu::destroy_webgpu_context(context);
  return result;
}

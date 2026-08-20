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
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/relu/relu_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sigmoid/sigmoid_wgsl.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr, config);
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

struct Conv1dRouteCase {
  const char* name;
  std::vector<uint32_t> input_dims;
  std::vector<uint32_t> weight_dims;
  std::vector<uint32_t> output_dims;
  int64_t stride;
  int64_t padding;
  int64_t dilation;
  int64_t groups;
  const char* expected_kernel;
};

void build_conv1d_route_graph(
    WebGPUGraph& graph,
    const Conv1dRouteCase& test_case) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  auto add_tensor = [&](const std::vector<uint32_t>& dims, int mem_obj_id) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb,
            vk::VkDataType::FLOAT32,
            &dims,
            /*constant_id=*/-1,
            mem_obj_id)
            .Union()));
    return id;
  };
  auto add_int = [&](int64_t value) {
    const int id = static_cast<int>(values.size());
    values.push_back(vk::CreateVkValue(
        fbb, vk::GraphTypes::Int, vk::CreateInt(fbb, value).Union()));
    return id;
  };
  auto add_int_list = [&](int64_t value) {
    const int id = static_cast<int>(values.size());
    const std::vector<int64_t> items = {value};
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::IntList,
        vk::CreateIntListDirect(fbb, &items).Union()));
    return id;
  };

  const int input = add_tensor(test_case.input_dims, 0);
  const int weight = add_tensor(test_case.weight_dims, 1);
  const int bias = static_cast<int>(values.size());
  values.push_back(vk::CreateVkValue(fbb));
  const int stride = add_int_list(test_case.stride);
  const int padding = add_int_list(test_case.padding);
  const int dilation = add_int_list(test_case.dilation);
  const int transposed = static_cast<int>(values.size());
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::Bool, vk::CreateBool(fbb, false).Union()));
  const int output_padding = add_int_list(0);
  const int groups = add_int(test_case.groups);
  const int output = add_tensor(test_case.output_dims, 2);
  const std::vector<int32_t> args = {
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(
      vk::CreateOperatorCallDirect(fbb, 0, "aten.convolution.default", &args));
  const std::vector<uint32_t> input_ids = {
      static_cast<uint32_t>(input), static_cast<uint32_t>(weight)};
  const std::vector<uint32_t> output_ids = {static_cast<uint32_t>(output)};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  graph.set_device(g_device);
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
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

TEST(WebGPUConv1dRoute, SelectsPointwiseGeneralAndSingleChannelDepthwise) {
  const Conv1dRouteCase cases[] = {
      {"pointwise", {1, 4, 8}, {6, 4, 1}, {1, 6, 8}, 1, 0, 1, 1, "conv1d_pw"},
      {"general", {1, 4, 10}, {6, 4, 3}, {1, 6, 4}, 2, 0, 1, 1, "conv1d"},
      {"single-channel depthwise",
       {1, 1, 8},
       {1, 1, 3},
       {1, 1, 8},
       1,
       1,
       1,
       1,
       "conv1d_dw"},
  };
  for (const Conv1dRouteCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    WebGPUGraph graph;
    build_conv1d_route_graph(graph, test_case);
    ASSERT_EQ(graph.num_dispatches(), 1);
    EXPECT_EQ(graph.dispatch_at(0).kernel_name, test_case.expected_kernel);
  }
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

constexpr int kResizeQ = 0;
constexpr int kResizeK = 1;
constexpr int kResizeUnrelated = 2;
constexpr int kResizeCascade = 3;
constexpr int kResizeOutput = 4;
constexpr int kResizeSymInt = 5;

void build_resize_test_graph(WebGPUGraph& graph) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  const std::vector<uint32_t> dims = {8, 8};
  for (int mem_obj_id = 0; mem_obj_id < 5; mem_obj_id++) {
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb,
            vk::VkDataType::FLOAT32,
            &dims,
            /*constant_id=*/-1,
            mem_obj_id)
            .Union()));
  }
  values.push_back(vk::CreateVkValue(
      fbb, vk::GraphTypes::SymInt, vk::CreateSymInt(fbb, 0).Union()));

  const std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  const std::vector<uint32_t> input_ids = {
      kResizeQ, kResizeK, kResizeUnrelated};
  const std::vector<uint32_t> output_ids = {kResizeOutput};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  graph.set_device(g_device);
  graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
}

WebGPUComputeDispatchDescriptor make_dynamic_test_descriptor(
    WebGPUGraph& graph,
    const char* kernel_name) {
  const auto& input = graph.get_tensor(kResizeQ);
  const auto& output = graph.get_tensor(kResizeOutput);
  WGPUBuffer params =
      graph.create_params_buffer(UnaryParams{64u, -1.0f, 1.0f, 0u});
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = "clamp";
  descriptor.kernel_name = kernel_name;
  descriptor.bindings = {
      {input.buffer, 0u, input.nbytes},
      {output.buffer, 0u, output.nbytes},
      {params, 0u, sizeof(UnaryParams)}};
  descriptor.constants = {{"wg_size", 64.0}};
  descriptor.grid = {99u, 97u};
  return descriptor;
}

struct ResizeProbeContext {
  int marker;
  int* observed;
  int* calls;
};

void record_resize_probe(WebGPUGraph&, const ResizeProbeContext& context) {
  *context.observed = context.marker;
  ++*context.calls;
}

struct ThrowingResizeContext {
  bool* fail;
  int* calls;
};

void maybe_throw_resize(WebGPUGraph&, const ThrowingResizeContext& context) {
  ++*context.calls;
  if (*context.fail) {
    throw std::runtime_error("resize hook failure");
  }
}

struct GridPickerContext {
  int tensor_id;
  uint32_t x_bias;
  uint32_t y_bias;
  int* calls;
  const bool* fail;
};

WebGPUDispatchGrid pick_tensor_grid(
    const WebGPUGraph& graph,
    const GridPickerContext& context) {
  ++*context.calls;
  if (context.fail != nullptr && *context.fail) {
    throw std::runtime_error("dynamic grid picker failure");
  }
  const auto& dims = graph.cur_dims(context.tensor_id);
  return {
      static_cast<uint32_t>(dims.at(0)) + context.x_bias,
      static_cast<uint32_t>(dims.at(1)) + context.y_bias};
}

struct CascadeContext {
  int source_id;
  int output_id;
};

void resize_cascade_output(WebGPUGraph& graph, const CascadeContext& context) {
  graph.set_cur_dims(context.output_id, graph.cur_dims(context.source_id));
}

void expect_dispatch_grid(
    WebGPUGraph& graph,
    size_t dispatch_index,
    uint32_t x,
    uint32_t y) {
  const auto& dispatch = graph.dispatch_at(dispatch_index);
  EXPECT_EQ(dispatch.workgroup_count_x, x);
  EXPECT_EQ(dispatch.workgroup_count_y, y);
}

TEST(WebGPUResizeHooks, TypedRegistrationOwnsContextCopies) {
  WebGPUGraph graph;
  build_resize_test_graph(graph);
  int tensor_observed = 0;
  int tensor_calls = 0;
  int symint_observed = 0;
  int symint_calls = 0;
  using ResizeProbeFn = void (*)(WebGPUGraph&, const ResizeProbeContext&);
  EXPECT_THROW(
      graph.add_tensor_resize_hook(
          kResizeQ,
          static_cast<ResizeProbeFn>(nullptr),
          ResizeProbeContext{0, &tensor_observed, &tensor_calls}),
      std::runtime_error);
  EXPECT_THROW(
      graph.add_tensor_resize_hook(
          kResizeSymInt,
          record_resize_probe,
          ResizeProbeContext{0, &tensor_observed, &tensor_calls}),
      std::runtime_error);
  EXPECT_THROW(
      graph.add_resize_hook(
          kResizeQ,
          record_resize_probe,
          ResizeProbeContext{0, &symint_observed, &symint_calls}),
      std::runtime_error);
  {
    ResizeProbeContext tensor_context = {17, &tensor_observed, &tensor_calls};
    ResizeProbeContext symint_context = {23, &symint_observed, &symint_calls};
    graph.add_tensor_resize_hook(kResizeQ, record_resize_probe, tensor_context);
    graph.add_resize_hook(kResizeSymInt, record_resize_probe, symint_context);
    tensor_context.marker = 101;
    symint_context.marker = 103;
  }

  graph.resize_input(kResizeQ, {7, 8});
  graph.propagate_resize();
  EXPECT_EQ(tensor_observed, 17);
  EXPECT_EQ(tensor_calls, 1);
  EXPECT_EQ(symint_observed, 0);
  EXPECT_EQ(symint_calls, 0);

  graph.set_symint(kResizeSymInt, 9);
  graph.propagate_resize();
  EXPECT_EQ(tensor_calls, 1);
  EXPECT_EQ(symint_observed, 23);
  EXPECT_EQ(symint_calls, 1);
}

TEST(WebGPUResizeHooks, RestoresDirtyTriggerWhenHookThrows) {
  WebGPUGraph graph;
  build_resize_test_graph(graph);
  bool hook_fails = true;
  int hook_calls = 0;
  int picker_calls = 0;
  graph.add_tensor_resize_hook(
      kResizeQ,
      maybe_throw_resize,
      ThrowingResizeContext{&hook_fails, &hook_calls});
  const size_t dispatch = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_hook_retry"),
      kResizeQ,
      pick_tensor_grid,
      GridPickerContext{kResizeQ, 1u, 2u, &picker_calls, nullptr});
  picker_calls = 0;

  graph.resize_input(kResizeQ, {4, 3});
  EXPECT_THROW(graph.propagate_resize(), std::runtime_error);
  EXPECT_EQ(hook_calls, 1);
  EXPECT_EQ(picker_calls, 0);
  expect_dispatch_grid(graph, dispatch, 9u, 10u);

  hook_fails = false;
  EXPECT_NO_THROW(graph.propagate_resize());
  EXPECT_EQ(hook_calls, 2);
  EXPECT_EQ(picker_calls, 1);
  expect_dispatch_grid(graph, dispatch, 5u, 5u);
}

TEST(WebGPUDynamicDispatch, InitializesAndIsolatesTriggeredGrids) {
  WebGPUGraph graph;
  build_resize_test_graph(graph);
  int q_calls = 0;
  int k_calls = 0;
  GridPickerContext q_context = {kResizeQ, 1u, 2u, &q_calls, nullptr};
  const GridPickerContext k_context = {kResizeK, 3u, 4u, &k_calls, nullptr};
  const size_t q_dispatch = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_q"),
      kResizeQ,
      pick_tensor_grid,
      q_context);
  const size_t k_dispatch = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_k"),
      kResizeK,
      pick_tensor_grid,
      k_context);
  q_context.x_bias = 101u;
  q_context.y_bias = 103u;
  expect_dispatch_grid(graph, q_dispatch, 9u, 10u);
  expect_dispatch_grid(graph, k_dispatch, 11u, 12u);
  EXPECT_EQ(q_calls, 1);
  EXPECT_EQ(k_calls, 1);
  q_calls = 0;
  k_calls = 0;

  graph.resize_input(kResizeUnrelated, {7, 7});
  graph.propagate_resize();
  EXPECT_EQ(q_calls, 0);
  EXPECT_EQ(k_calls, 0);

  graph.resize_input(kResizeQ, {4, 3});
  graph.propagate_resize();
  expect_dispatch_grid(graph, q_dispatch, 5u, 5u);
  expect_dispatch_grid(graph, k_dispatch, 11u, 12u);
  EXPECT_EQ(q_calls, 1);
  EXPECT_EQ(k_calls, 0);

  graph.resize_input(kResizeQ, {2, 5});
  graph.propagate_resize();
  expect_dispatch_grid(graph, q_dispatch, 3u, 7u);
  EXPECT_EQ(q_calls, 2);
  graph.resize_input(kResizeQ, {2, 5});
  graph.propagate_resize();
  EXPECT_EQ(q_calls, 2);

  graph.resize_input(kResizeK, {6, 1});
  graph.propagate_resize();
  expect_dispatch_grid(graph, k_dispatch, 9u, 5u);
  EXPECT_EQ(k_calls, 1);
}

TEST(WebGPUDynamicDispatch, HandlesCascadesAndStagesPickerFailures) {
  {
    WebGPUGraph graph;
    build_resize_test_graph(graph);
    int same_pass_calls = 0;
    int cascade_pass_calls = 0;
    graph.add_tensor_resize_hook(
        kResizeQ,
        resize_cascade_output,
        CascadeContext{kResizeQ, kResizeCascade});
    const size_t same_pass_dispatch = graph.add_dynamic_compute_dispatch(
        make_dynamic_test_descriptor(graph, "dynamic_same_pass"),
        kResizeQ,
        pick_tensor_grid,
        GridPickerContext{kResizeCascade, 5u, 7u, &same_pass_calls, nullptr});
    const size_t cascade_pass_dispatch = graph.add_dynamic_compute_dispatch(
        make_dynamic_test_descriptor(graph, "dynamic_cascade_pass"),
        kResizeCascade,
        pick_tensor_grid,
        GridPickerContext{
            kResizeCascade, 9u, 11u, &cascade_pass_calls, nullptr});
    same_pass_calls = 0;
    cascade_pass_calls = 0;
    graph.resize_input(kResizeQ, {4, 6});
    graph.propagate_resize();
    EXPECT_EQ(same_pass_calls, 1);
    EXPECT_EQ(cascade_pass_calls, 1);
    expect_dispatch_grid(graph, same_pass_dispatch, 9u, 13u);
    expect_dispatch_grid(graph, cascade_pass_dispatch, 13u, 17u);
  }

  WebGPUGraph graph;
  build_resize_test_graph(graph);
  int first_calls = 0;
  int second_calls = 0;
  bool second_fails = false;
  const size_t first = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_first"),
      kResizeQ,
      pick_tensor_grid,
      GridPickerContext{kResizeQ, 1u, 2u, &first_calls, nullptr});
  const size_t second = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_second"),
      kResizeQ,
      pick_tensor_grid,
      GridPickerContext{kResizeQ, 3u, 4u, &second_calls, &second_fails});
  expect_dispatch_grid(graph, first, 9u, 10u);
  expect_dispatch_grid(graph, second, 11u, 12u);
  second_fails = true;
  graph.resize_input(kResizeQ, {4, 3});
  EXPECT_THROW(graph.propagate_resize(), std::runtime_error);
  expect_dispatch_grid(graph, first, 9u, 10u);
  expect_dispatch_grid(graph, second, 11u, 12u);
  second_fails = false;
  EXPECT_NO_THROW(graph.propagate_resize());
  expect_dispatch_grid(graph, first, 5u, 5u);
  expect_dispatch_grid(graph, second, 7u, 7u);
}

TEST(WebGPUDynamicDispatch, RejectsRouteOverlapWithoutPoisoningRegistry) {
  WebGPUGraph graph;
  build_resize_test_graph(graph);
  int calls = 0;
  const size_t dispatches_before_invalid_trigger = graph.num_dispatches();
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_negative_trigger"),
          -1,
          pick_tensor_grid,
          GridPickerContext{kResizeQ, 0u, 0u, &calls, nullptr}),
      std::runtime_error);
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_oob_trigger"),
          graph.num_values(),
          pick_tensor_grid,
          GridPickerContext{kResizeQ, 0u, 0u, &calls, nullptr}),
      std::runtime_error);
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_symint_trigger"),
          kResizeSymInt,
          pick_tensor_grid,
          GridPickerContext{kResizeQ, 0u, 0u, &calls, nullptr}),
      std::runtime_error);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(graph.num_dispatches(), dispatches_before_invalid_trigger);

  auto pick_zero_grid = [](const WebGPUGraph&, const WebGPUDispatchGrid& grid) {
    return grid;
  };
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_zero_x"),
          kResizeQ,
          +pick_zero_grid,
          WebGPUDispatchGrid{0u, 1u}),
      std::runtime_error);
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_zero_y"),
          kResizeQ,
          +pick_zero_grid,
          WebGPUDispatchGrid{1u, 0u}),
      std::runtime_error);
  EXPECT_EQ(graph.num_dispatches(), dispatches_before_invalid_trigger);

  bool initial_fails = true;
  const size_t dispatches_before_failure = graph.num_dispatches();
  EXPECT_THROW(
      graph.add_dynamic_compute_dispatch(
          make_dynamic_test_descriptor(graph, "dynamic_initial_failure"),
          kResizeQ,
          pick_tensor_grid,
          GridPickerContext{kResizeQ, 0u, 0u, &calls, &initial_fails}),
      std::runtime_error);
  EXPECT_EQ(graph.num_dispatches(), dispatches_before_failure);

  const size_t dynamic = graph.add_dynamic_compute_dispatch(
      make_dynamic_test_descriptor(graph, "dynamic_route_guard"),
      kResizeQ,
      pick_tensor_grid,
      GridPickerContext{kResizeQ, 0u, 0u, &calls, nullptr});
  ASSERT_EQ(dynamic, 0u);
  graph.add_dispatch(WebGPUDispatch{});
  graph.add_dispatch(WebGPUDispatch{});

  EXPECT_THROW(
      graph.register_dispatch_route_group({{0, 1}, {1, 2}}),
      std::runtime_error);
  calls = 0;
  graph.resize_input(kResizeQ, {7, 6});
  graph.propagate_resize();
  EXPECT_EQ(calls, 1);
  expect_dispatch_grid(graph, dynamic, 7u, 6u);
  const size_t group = graph.register_dispatch_route_group({{1, 2}, {2, 3}});
  EXPECT_EQ(group, 0u);
  graph.select_dispatch_route(group, 1, {{13u, 17u}});
  expect_dispatch_grid(graph, 1u, 0u, 0u);
  expect_dispatch_grid(graph, 2u, 13u, 17u);
}

struct InvalidRopeGraphCase {
  const char* name;
  std::vector<uint32_t> xq_dims;
  std::vector<uint32_t> xk_dims;
  std::vector<uint32_t> cos_dims;
  std::vector<uint32_t> sin_dims;
  std::vector<uint32_t> xq_out_dims;
  std::vector<uint32_t> xk_out_dims;
  vkgraph::VkDataType xq_dtype;
  const char* expected_error;
};

void expect_invalid_rope_graph(const InvalidRopeGraphCase& test_case) {
  namespace vk = vkgraph;
  ::flatbuffers::FlatBufferBuilder fbb;
  std::vector<::flatbuffers::Offset<vk::VkValue>> values;
  auto add_tensor = [&](vk::VkDataType dtype,
                        const std::vector<uint32_t>& dims,
                        int mem_obj_id) {
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(
            fbb, dtype, &dims, /*constant_id=*/-1, mem_obj_id)
            .Union()));
  };
  add_tensor(test_case.xq_dtype, test_case.xq_dims, 0);
  add_tensor(vk::VkDataType::FLOAT32, test_case.xk_dims, 1);
  add_tensor(vk::VkDataType::FLOAT32, test_case.cos_dims, 2);
  add_tensor(vk::VkDataType::FLOAT32, test_case.sin_dims, 3);
  add_tensor(vk::VkDataType::FLOAT32, test_case.xq_out_dims, 4);
  add_tensor(vk::VkDataType::FLOAT32, test_case.xk_out_dims, 5);
  std::vector<int32_t> output_value_ids = {4, 5};
  values.push_back(vk::CreateVkValue(
      fbb,
      vk::GraphTypes::ValueList,
      vk::CreateValueListDirect(fbb, &output_value_ids).Union()));

  std::vector<int32_t> args = {0, 1, 2, 3, 6};
  std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
  chain.push_back(vk::CreateOperatorCallDirect(
      fbb, 0, "et_vk.apply_rotary_emb.default", &args));
  std::vector<uint32_t> input_ids = {0, 1, 2, 3};
  std::vector<uint32_t> output_ids = {4, 5};
  const auto root = vk::CreateVkGraphDirect(
      fbb, "0", &chain, &values, &input_ids, &output_ids);
  vk::FinishVkGraphBuffer(fbb, root);

  WebGPUGraph graph;
  std::string error;
  try {
    graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
  } catch (const std::exception& exception) {
    error = exception.what();
  }
  EXPECT_FALSE(error.empty()) << test_case.name << " unexpectedly built";
  EXPECT_EQ(error, test_case.expected_error)
      << test_case.name << " rejected for the wrong reason";
  const WebGPUMemoryStats stats = graph.memory_stats();
  EXPECT_EQ(stats.num_dispatches, 0) << test_case.name;
  EXPECT_EQ(stats.uniform_buffer_bytes, 0u) << test_case.name;
  EXPECT_EQ(stats.num_cached_shaders, 0) << test_case.name;
  EXPECT_EQ(stats.num_cached_pipelines, 0) << test_case.name;
}

TEST(WebGPURopeValidation, RejectsMalformedGraphsBeforeDispatchAllocation) {
  ASSERT_TRUE(
      webgpu_operator_registry().has_op("et_vk.apply_rotary_emb.default"));
  const std::vector<uint32_t> xq = {1, 2, 2, 4};
  const std::vector<uint32_t> xk = {1, 2, 1, 4};
  const std::vector<uint32_t> freqs = {2, 2};
  const InvalidRopeGraphCase cases[] = {
      {"query rank",
       {2, 4},
       xk,
       freqs,
       freqs,
       {2, 4},
       xk,
       vkgraph::VkDataType::FLOAT32,
       "WebGPU apply_rotary_emb: malformed dims"},
      {"sequence mismatch",
       xq,
       {1, 3, 1, 4},
       freqs,
       freqs,
       xq,
       {1, 3, 1, 4},
       vkgraph::VkDataType::FLOAT32,
       "WebGPU apply_rotary_emb: xq/xk head_dim and seq must match"},
      {"head dimension mismatch",
       xq,
       {1, 2, 1, 6},
       freqs,
       freqs,
       xq,
       {1, 2, 1, 6},
       vkgraph::VkDataType::FLOAT32,
       "WebGPU apply_rotary_emb: xq/xk head_dim and seq must match"},
      {"frequency width mismatch",
       xq,
       xk,
       {2, 3},
       {2, 3},
       xq,
       xk,
       vkgraph::VkDataType::FLOAT32,
       "WebGPU apply_rotary_emb: head_dim != 2 * freqs_cos last dim"},
      {"cosine/sine shape mismatch",
       xq,
       xk,
       freqs,
       {2, 1},
       xq,
       xk,
       vkgraph::VkDataType::FLOAT32,
       "WebGPU apply_rotary_emb: freqs_cos and freqs_sin shapes differ"},
      {"query byte size mismatch",
       xq,
       xk,
       freqs,
       freqs,
       xq,
       xk,
       vkgraph::VkDataType::INT64,
       "WebGPU apply_rotary_emb: dtype/byte-size mismatch (all fp32) or "
       "freqs shape != [seq, head_dim/2]"},
  };
  for (const InvalidRopeGraphCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    expect_invalid_rope_graph(test_case);
  }
}

TEST(WebGPUToCopyValidation, RejectsBoolAndByteIntegerConversions) {
  ASSERT_TRUE(webgpu_operator_registry().has_op("aten._to_copy.default"));
  namespace vk = vkgraph;
  struct TestCase {
    const char* name;
    vk::VkDataType input_dtype;
    vk::VkDataType output_dtype;
  };
  const TestCase cases[] = {
      {"bool_to_int8", vk::VkDataType::BOOL, vk::VkDataType::INT8},
      {"bool_to_uint8", vk::VkDataType::BOOL, vk::VkDataType::UINT8},
      {"int8_to_bool", vk::VkDataType::INT8, vk::VkDataType::BOOL},
      {"uint8_to_bool", vk::VkDataType::UINT8, vk::VkDataType::BOOL},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ::flatbuffers::FlatBufferBuilder fbb;
    const std::vector<uint32_t> dims = {4};
    std::vector<::flatbuffers::Offset<vk::VkValue>> values;
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(fbb, test_case.input_dtype, &dims, -1, 0)
            .Union()));
    values.push_back(vk::CreateVkValue(
        fbb,
        vk::GraphTypes::VkTensor,
        vk::CreateVkTensorDirect(fbb, test_case.output_dtype, &dims, -1, 1)
            .Union()));
    const std::vector<int32_t> args = {0, 1};
    std::vector<::flatbuffers::Offset<vk::OperatorCall>> chain;
    chain.push_back(
        vk::CreateOperatorCallDirect(fbb, 0, "aten._to_copy.default", &args));
    const std::vector<uint32_t> input_ids = {0};
    const std::vector<uint32_t> output_ids = {1};
    const auto root = vk::CreateVkGraphDirect(
        fbb, "0", &chain, &values, &input_ids, &output_ids);
    vk::FinishVkGraphBuffer(fbb, root);

    WebGPUGraph graph;
    try {
      graph.build(fbb.GetBufferPointer(), nullptr, 0, nullptr);
      FAIL() << test_case.name << " unexpectedly built";
    } catch (const std::runtime_error& error) {
      EXPECT_STREQ(
          error.what(),
          "WebGPU to_copy: bool and integer conversions are unsupported");
    }
    EXPECT_EQ(graph.memory_stats().num_dispatches, 0);
  }
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
  std::vector<OutputData> outputs;

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

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {
namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
WGPUDevice g_device = nullptr;
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

struct SigmoidParams {
  uint32_t num_elements;
  uint32_t padding[3];
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

TEST(WebGPUShaderRegistry, FindsKnownShaderAndRejectsUnknownName) {
  const WebGPUShaderInfo& sigmoid = get_webgpu_shader_info("sigmoid");
  EXPECT_EQ(sigmoid.name, "sigmoid");
  EXPECT_NE(sigmoid.source, nullptr);
  EXPECT_GT(sigmoid.workgroup_size_x, 0u);
  EXPECT_THROW(
      get_webgpu_shader_info("not_a_registered_shader"), std::runtime_error);
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
      WGPUBuffer params =
          graph.create_params_buffer(SigmoidParams{kNumElements, {0, 0, 0}});

      WebGPUComputeDispatchDescriptor descriptor;
      descriptor.shader_name = "sigmoid";
      descriptor.kernel_name = "sigmoid_test";
      descriptor.bindings = {
          {input, 0u, kBufferBytes},
          {output, 0u, kBufferBytes},
          {params, 0u, sizeof(SigmoidParams)}};
      descriptor.constants = {{"wg_size", 64.0}};
      descriptor.grid = {1u, 1u};

      graph.add_compute_dispatch(descriptor);
      graph.add_compute_dispatch(descriptor);

      const WebGPUMemoryStats stats = graph.memory_stats();
      EXPECT_EQ(stats.num_dispatches, 2);
      EXPECT_EQ(stats.num_cached_shaders, 1);
      EXPECT_EQ(stats.num_cached_pipelines, 1);
    }
    wgpuBufferRelease(output);
    wgpuBufferRelease(input);
  }
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

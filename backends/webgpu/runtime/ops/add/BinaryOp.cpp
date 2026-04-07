/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/add/binary_add_wgsl.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstring>

namespace executorch {
namespace backends {
namespace webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct.
// Must be 16-byte aligned for WebGPU uniform buffer requirements.
struct AddParams {
  uint32_t num_elements;
  float alpha;
  uint32_t _pad[2]; // pad to 16 bytes
};

void add_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.add.Tensor args: [in1, in2, alpha, out]
  const int in1_id = args.at(0);
  const int in2_id = args.at(1);
  const int alpha_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  // Get alpha value (defaults to 1.0 if not a scalar)
  float alpha = 1.0f;
  if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Int) {
    alpha = static_cast<float>(graph.get_int(alpha_id));
  } else if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Double) {
    alpha = static_cast<float>(graph.get_double(alpha_id));
  }

  const auto& out_tensor = graph.get_tensor(out_id);
  uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  // Create uniform buffer for params
  AddParams params = {};
  params.num_elements = num_elements;
  params.alpha = alpha;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(AddParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped = wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(AddParams));
  std::memcpy(mapped, &params, sizeof(AddParams));
  wgpuBufferUnmap(uniform_buffer);

  // Create shader module from built-in WGSL source
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kBinaryAddWGSL, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Create bind group layout: 3 storage buffers + 1 uniform
  WGPUBindGroupLayoutEntry entries[4] = {};

  // input1 - storage buffer, read-only
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  // input2 - storage buffer, read-only
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  // output - storage buffer, read-write
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;

  // params - uniform buffer
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl =
      wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  // Create pipeline layout
  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  // Create compute pipeline
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  // Create bind group with actual buffers
  const auto& in1_tensor = graph.get_tensor(in1_id);
  const auto& in2_tensor = graph.get_tensor(in2_id);

  WGPUBindGroupEntry bg_entries[4] = {};

  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in1_tensor.buffer;
  bg_entries[0].size = in1_tensor.nbytes;

  bg_entries[1].binding = 1;
  bg_entries[1].buffer = in2_tensor.buffer;
  bg_entries[1].size = in2_tensor.nbytes;

  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out_tensor.buffer;
  bg_entries[2].size = out_tensor.nbytes;

  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(AddParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  uint32_t workgroup_count =
      (num_elements + kBinaryAddWorkgroupSize - 1) / kBinaryAddWorkgroupSize;

  graph.add_dispatch({pipeline, bind_group, workgroup_count});

  // Release intermediate objects (pipeline and bind_group are kept by dispatch)
  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // uniform_buffer is kept alive by the bind group
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.add.Tensor, add_impl);
}

} // namespace webgpu
} // namespace backends
} // namespace executorch

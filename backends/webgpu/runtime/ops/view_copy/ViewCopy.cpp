/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy.h>
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy_wgsl.h>

#include <webgpu/webgpu.h>

#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct; 16-byte aligned.
struct ViewCopyParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

} // namespace

void add_flat_copy(WebGPUGraph& graph, int in_id, int out_id) {
  // get_tensor doesn't type-check; assert both args are tensors (fail loud).
  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("flat_copy: in/out arg is not a tensor");
  }

  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  // Flat byte copy assumes dense row-major operands; the WebGPU buffer
  // backend only produces contiguous tensors, so a strided/transposed
  // view cannot reach here.

  // 4-byte (fp32) alignment guard on both operands; also the dtype guard.
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("flat_copy: operand not 4-byte aligned");
  }

  const uint32_t in_numel =
      static_cast<uint32_t>(in_tensor.nbytes / sizeof(float));
  const uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  // view preserves numel; this guard also prevents an OOB input[] read.
  if (in_numel != num_elements) {
    throw std::runtime_error("flat_copy: input/output element count mismatch");
  }

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kViewCopyWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, num_elements, wg_size, "view_copy");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ViewCopyParams params = {};
  params.num_elements = num_elements;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(ViewCopyParams));
  graph.add_uniform_buffer_bytes(sizeof(ViewCopyParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kViewCopyWGSL, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group: read storage (input) + read_write storage (output) + uniform.
  WGPUBindGroupLayoutEntry entries[3] = {};

  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;

  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 3;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[3] = {};

  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;

  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;

  bg_entries[2].binding = 2;
  bg_entries[2].buffer = uniform_buffer;
  bg_entries[2].size = sizeof(ViewCopyParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Drop our ref; the bind group keeps the uniform buffer alive until release.
  wgpuBufferRelease(uniform_buffer);
}

namespace {

// view_copy = contiguous reshape = flat copy (mirrors Vulkan view_buffer).
void view_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, size, out]; out = last value-id (shape from out_tensor.dims).
  add_flat_copy(graph, args.at(0), args.at(args.size() - 1));
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.view_copy.default, view_copy_impl);
}

} // namespace executorch::backends::webgpu

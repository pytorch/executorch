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
#include <executorch/backends/webgpu/runtime/ops/fill/fill_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct (16-byte aligned).
struct FillParams {
  uint32_t num_elements;
  float fill_value;
  uint32_t _pad[2];
};

float read_scalar(WebGPUGraph& graph, int id, const char* op_name) {
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::Double) {
    return static_cast<float>(graph.get_double(id));
  }
  if (graph.get_value_type(id) == WebGPUGraph::ValueType::Int) {
    return static_cast<float>(graph.get_int(id));
  }
  throw std::runtime_error(
      std::string(op_name) + ": fill value is not a scalar");
}

// Fills the (pre-allocated) output buffer with a constant scalar.
void add_fill(
    WebGPUGraph& graph,
    int out_id,
    float fill_value,
    const char* op_name) {
  WGPUDevice device = graph.device();
  const auto& out_tensor = graph.get_tensor(out_id);
  if (out_tensor.buffer == nullptr) {
    throw std::runtime_error(std::string(op_name) + ": null output buffer");
  }
  uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  uint32_t wg_size = utils::clamp_workgroup_size(device, kFillWorkgroupSizeX);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, num_elements, wg_size, op_name);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  FillParams params = {};
  params.num_elements = num_elements;
  params.fill_value = fill_value;
  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(FillParams));
  graph.add_uniform_buffer_bytes(sizeof(FillParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kFillWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[2] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 2;
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

  WGPUBindGroupEntry bg_entries[2] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out_tensor.buffer;
  bg_entries[0].size = out_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = uniform_buffer;
  bg_entries[1].size = sizeof(FillParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 2;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline, bind_group, workgroup_count.x, "", workgroup_count.y});

  // Dynamic shapes: recompute num_elements/dispatch from the live output dims.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      out_id,
      [out_id, fill_value, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(out_id);
        const uint64_t numel = utils::numel_of(d);
        FillParams p = {};
        p.num_elements = static_cast<uint32_t>(numel);
        p.fill_value = fill_value;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), static_cast<uint32_t>(numel), wg_size, "fill(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

void full_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_fill(
      graph,
      args.at(args.size() - 1),
      read_scalar(graph, args.at(1), "full"),
      "full");
}

void full_like_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_fill(
      graph,
      args.at(args.size() - 1),
      read_scalar(graph, args.at(1), "full_like"),
      "full_like");
}

void scalar_tensor_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  add_fill(
      graph,
      args.at(args.size() - 1),
      read_scalar(graph, args.at(0), "scalar_tensor"),
      "scalar_tensor");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.full.default, full_impl);
  WEBGPU_REGISTER_OP(aten.full_like.default, full_like_impl);
  WEBGPU_REGISTER_OP(aten.scalar_tensor.default, scalar_tensor_impl);
}

} // namespace executorch::backends::webgpu

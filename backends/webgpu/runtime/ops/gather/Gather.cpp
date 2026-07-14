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
#include <executorch/backends/webgpu/runtime/ops/TensorMeta.h>
#include <executorch/backends/webgpu/runtime/ops/gather/gather_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct GatherParams {
  uint32_t dim;
  uint32_t _pad[3];
};

// gather: out[c] = self[c] with c[dim] replaced by index[c].
void gather_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int self_id = args.at(0);
  const int dim_id = args.at(1);
  const int index_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("gather: dim is not an int");
  }
  WGPUDevice device = graph.device();
  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& index_tensor = graph.get_tensor(index_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (self_tensor.buffer == nullptr || index_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("gather: null buffer binding");
  }

  const int64_t ndim = static_cast<int64_t>(out_tensor.dims.size());
  int64_t dim = graph.get_int(dim_id);
  if (dim < 0) {
    dim += ndim;
  }
  if (ndim == 0 || dim < 0 || dim >= ndim) {
    throw std::runtime_error("gather: dim out of range");
  }

  TensorMeta out_meta;
  TensorMeta self_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(self_tensor, &self_meta);
  if (out_meta.ndim != self_meta.ndim) {
    throw std::runtime_error("gather: self/out rank mismatch");
  }

  const size_t out_numel = out_tensor.nbytes / sizeof(float);
  const size_t index_numel = index_tensor.nbytes / sizeof(int32_t);
  if (out_tensor.nbytes != out_numel * sizeof(float) ||
      self_tensor.nbytes % sizeof(float) != 0 ||
      index_tensor.nbytes != index_numel * sizeof(int32_t)) {
    throw std::runtime_error("gather: fp32 self/out + i32 index required");
  }
  if (out_numel != index_numel) {
    throw std::runtime_error("gather: out numel != index numel");
  }

  uint32_t wg_size = utils::clamp_workgroup_size(device, kGatherWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, out_meta.numel, wg_size, "gather");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  GatherParams params = {};
  params.dim = static_cast<uint32_t>(dim);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer self_meta_buf =
      utils::make_uniform(device, &self_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(GatherParams));
  graph.add_uniform_buffer_bytes(2 * sizeof(TensorMeta) + sizeof(GatherParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kGatherWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[6] = {};
  entries[0].binding = 0;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].binding = 2;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;
  entries[3].binding = 3;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  entries[4].binding = 4;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;
  entries[5].binding = 5;
  entries[5].buffer.type = WGPUBufferBindingType_Uniform;
  for (auto& e : entries) {
    e.visibility = WGPUShaderStage_Compute;
  }

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 6;
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

  WGPUBindGroupEntry bg_entries[6] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = self_tensor.buffer;
  bg_entries[0].size = self_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = index_tensor.buffer;
  bg_entries[1].size = index_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out_tensor.buffer;
  bg_entries[2].size = out_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = out_meta_buf;
  bg_entries[3].size = sizeof(TensorMeta);
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = self_meta_buf;
  bg_entries[4].size = sizeof(TensorMeta);
  bg_entries[5].binding = 5;
  bg_entries[5].buffer = params_buf;
  bg_entries[5].size = sizeof(GatherParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(self_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.gather.default, gather_impl);
}

} // namespace executorch::backends::webgpu

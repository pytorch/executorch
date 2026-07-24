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
#include <executorch/backends/webgpu/runtime/ops/index_select/index_select_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct IndexSelectParams {
  uint32_t info[4]; // info[0] = dim
};
static_assert(
    sizeof(IndexSelectParams) == 16,
    "IndexSelectParams must match the WGSL Params vec4<u32> (16 bytes)");

// index_select: gather rows along dim via an int index (Vulkan IndexSelect).
void index_select_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, dim, index, out]; index is an int32 tensor (downcast_64_bit).
  const int self_id = args.at(0);
  const int dim_id = args.at(1);
  const int index_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(self_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(index_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("index_select: self/index/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& index_tensor = graph.get_tensor(index_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (self_tensor.buffer == nullptr || index_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("index_select: null buffer binding");
  }

  const int64_t ndim = static_cast<int64_t>(self_tensor.dims.size());
  if (ndim > static_cast<int64_t>(kTensorMetaMaxNdim)) {
    throw std::runtime_error("index_select: tensor rank exceeds 8 (MAX_NDIM)");
  }

  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("index_select: dim arg is not a static Int");
  }
  int64_t dim = graph.get_int(dim_id);
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw std::runtime_error("index_select: dim out of range");
  }

  TensorMeta out_meta;
  TensorMeta in_meta;
  fill_tensor_meta(out_tensor, &out_meta);
  fill_tensor_meta(self_tensor, &in_meta);
  if (out_tensor.nbytes !=
          static_cast<size_t>(out_meta.numel) * sizeof(float) ||
      self_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("index_select: non-fp32 self/out (nbytes % 4)");
  }
  // Index is the int32 downcast of the int64 index (downcast_64_bit).
  if (index_tensor.nbytes % sizeof(int32_t) != 0) {
    throw std::runtime_error("index_select: index buffer is not int32");
  }
  // The index gathers out.dims[dim] rows (one per index element), so it must
  // hold at least that many entries.
  uint64_t index_numel = 1;
  for (int64_t d : index_tensor.dims) {
    index_numel *= static_cast<uint64_t>(d);
  }
  if (index_numel < static_cast<uint64_t>(out_tensor.dims.at(dim))) {
    throw std::runtime_error("index_select: index numel < out.dims[dim]");
  }

  IndexSelectParams params = {};
  params.info[0] = static_cast<uint32_t>(dim);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kIndexSelectWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, out_meta.numel, wg_size, "index_select");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer out_meta_buf =
      utils::make_uniform(device, &out_meta, sizeof(TensorMeta));
  WGPUBuffer in_meta_buf =
      utils::make_uniform(device, &in_meta, sizeof(TensorMeta));
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(IndexSelectParams));
  graph.add_uniform_buffer_bytes(
      2 * sizeof(TensorMeta) + sizeof(IndexSelectParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kIndexSelectWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // in, out (rw), index (read i32), out_meta, in_meta, params (3 uniforms).
  WGPUBindGroupLayoutEntry entries[6] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;
  entries[4].binding = 4;
  entries[4].visibility = WGPUShaderStage_Compute;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;
  entries[5].binding = 5;
  entries[5].visibility = WGPUShaderStage_Compute;
  entries[5].buffer.type = WGPUBufferBindingType_Uniform;

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
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = index_tensor.buffer;
  bg_entries[2].size = index_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = out_meta_buf;
  bg_entries[3].size = sizeof(TensorMeta);
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = in_meta_buf;
  bg_entries[4].size = sizeof(TensorMeta);
  bg_entries[5].binding = 5;
  bg_entries[5].buffer = params_buf;
  bg_entries[5].size = sizeof(IndexSelectParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  // Static shapes only: index_select registers no resize hook, so the output
  // extent (out.dims[dim] == index numel) is fixed at build time.
  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "index_select",
       workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Drop our refs; the bind group keeps the uniforms alive until release.
  wgpuBufferRelease(out_meta_buf);
  wgpuBufferRelease(in_meta_buf);
  wgpuBufferRelease(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.index_select.default, index_select_impl);
}

} // namespace executorch::backends::webgpu

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
#include <executorch/backends/webgpu/runtime/ops/index/index_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct IndexParams {
  uint32_t numel;
  uint32_t _pad[3]; // pad to 16 bytes
};

// aten.index.Tensor 1D-self gather out[i]=self[index[i]] (mirrors Vulkan).
void index_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, indices (Tensor?[] -> ValueList), out].
  const int self_id = args.at(0);
  const int list_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(self_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("index: self arg is not a tensor");
  }
  if (graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("index: out arg is not a tensor");
  }
  if (graph.get_value_type(list_id) != WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error("index: indices arg is not a ValueList");
  }

  // Exactly one non-Null index tensor (mirror Vulkan IndexTensor.cpp:67-69).
  const std::vector<int>& ids = graph.get_value_list(list_id);
  int index_id = -1;
  for (int id : ids) {
    if (graph.get_value_type(id) == WebGPUGraph::ValueType::Null) {
      continue;
    }
    if (graph.get_value_type(id) != WebGPUGraph::ValueType::Tensor) {
      throw std::runtime_error("index: index list element is not a tensor");
    }
    if (index_id != -1) {
      throw std::runtime_error("index: expected exactly one index tensor");
    }
    index_id = id;
  }
  if (index_id == -1) {
    throw std::runtime_error("index: no index tensor provided");
  }

  WGPUDevice device = graph.device();

  const auto& self_tensor = graph.get_tensor(self_id);
  const auto& index_tensor = graph.get_tensor(index_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (self_tensor.buffer == nullptr || index_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("index: null buffer binding");
  }
  // 1D-self gather: the kernel flat-indexes self by a scalar; fail loud on a
  // higher-rank self (mirrors Vulkan index_tensor_buffer's 1D-self contract).
  if (self_tensor.dims.size() != 1) {
    throw std::runtime_error("index: only 1D self is supported");
  }

  const size_t out_numel = out_tensor.nbytes / sizeof(float);
  if (out_tensor.nbytes != out_numel * sizeof(float) ||
      self_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("index: non-fp32 self/out (nbytes != numel * 4)");
  }
  // Index is the int32 downcast of the int64 advanced index (downcast_64_bit).
  const size_t index_numel = index_tensor.nbytes / sizeof(int32_t);
  if (index_tensor.nbytes != index_numel * sizeof(int32_t)) {
    throw std::runtime_error("index: index buffer is not int32 (nbytes % 4)");
  }
  // out is one self element per index element (row_width == 1, 1D self).
  if (out_numel != index_numel) {
    throw std::runtime_error("index: out numel != index numel");
  }

  uint32_t num_elements = static_cast<uint32_t>(out_numel);
  uint32_t wg_size = utils::clamp_workgroup_size(device, kIndexWorkgroupSizeX);
  uint32_t workgroup_count =
      utils::compute_1d_workgroup_count(device, num_elements, wg_size, "index");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  IndexParams params = {};
  params.numel = num_elements;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(IndexParams));
  graph.add_uniform_buffer_bytes(sizeof(IndexParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kIndexWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // self (read), out (read_write), index (read i32), params (uniform).
  WGPUBindGroupLayoutEntry entries[4] = {};
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

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
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

  WGPUBindGroupEntry bg_entries[4] = {};
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
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(IndexParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // The bind group keeps the uniform buffer alive until release.
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.index.Tensor, index_impl);
}

} // namespace executorch::backends::webgpu

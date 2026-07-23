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
#include <executorch/backends/webgpu/runtime/ops/embedding/embedding_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct EmbeddingParams {
  uint32_t num_elements;
  uint32_t dim;
  uint32_t _pad[2];
};

// aten.embedding: out[row, :] = weight[indices[row], :] (fp32 weight, i32 idx).
void embedding_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int weight_id = args.at(0);
  const int indices_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();
  const auto& weight = graph.get_tensor(weight_id);
  const auto& indices = graph.get_tensor(indices_id);
  const auto& out = graph.get_tensor(out_id);

  if (weight.buffer == nullptr || indices.buffer == nullptr ||
      out.buffer == nullptr) {
    throw std::runtime_error("embedding: null buffer binding");
  }
  if (weight.dims.size() != 2) {
    throw std::runtime_error("embedding: weight must be 2D [vocab, dim]");
  }
  const uint32_t dim = static_cast<uint32_t>(weight.dims[1]);
  if (dim == 0) {
    throw std::runtime_error("embedding: dim == 0");
  }
  const size_t out_numel = out.nbytes / sizeof(float);
  // Index is the int32 downcast of the int64 indices (mirror index op).
  const size_t index_numel = indices.nbytes / sizeof(int32_t);
  if (out.nbytes != out_numel * sizeof(float) ||
      weight.nbytes % sizeof(float) != 0 ||
      indices.nbytes != index_numel * sizeof(int32_t)) {
    throw std::runtime_error(
        "embedding: fp32 weight/out + i32 indices required");
  }
  if (out_numel != index_numel * dim) {
    throw std::runtime_error("embedding: out numel != num_indices * dim");
  }

  uint32_t num_elements = static_cast<uint32_t>(out_numel);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kEmbeddingWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_elements, wg_size, "embedding");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  EmbeddingParams params = {};
  params.num_elements = num_elements;
  params.dim = dim;
  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(EmbeddingParams));
  graph.add_uniform_buffer_bytes(sizeof(EmbeddingParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kEmbeddingWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_Storage;
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
  bg_entries[0].buffer = weight.buffer;
  bg_entries[0].size = weight.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = indices.buffer;
  bg_entries[1].size = indices.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = out.buffer;
  bg_entries[2].size = out.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(EmbeddingParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline, bind_group, workgroup_count.x, "", workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.embedding.default, embedding_impl);
}

} // namespace executorch::backends::webgpu

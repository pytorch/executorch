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
#include <executorch/backends/webgpu/runtime/ops/embedding_q4gsw/embedding_q4gsw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned, 32 bytes).
struct EmbeddingParams {
  uint32_t embed_dim;
  uint32_t blocks_per_row;
  uint32_t num_indices;
  uint32_t group_size;
  uint32_t groups_per_row;
  uint32_t bytes_per_row;
  uint32_t total_blocks;
  uint32_t _pad;
};
static_assert(
    sizeof(EmbeddingParams) == 32,
    "EmbeddingParams must be 32 bytes");

// arg order mirrors Vulkan EmbeddingQ4gsw.cpp.
void embedding_q4gsw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int weight_id = args.at(0);
  const int scales_id = args.at(1);
  const int group_size_id = args.at(2);
  const int indices_id = args.at(3);
  const int is_linear_weight_id = args.at(4);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();

  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& indices = graph.get_tensor(indices_id);
  const auto& out = graph.get_tensor(out_id);

  // Only the flat weight path is supported (linear-block unsupported).
  bool is_linear = false;
  if (graph.get_value_type(is_linear_weight_id) ==
      WebGPUGraph::ValueType::Bool) {
    is_linear = graph.get_bool(is_linear_weight_id);
  } else if (
      graph.get_value_type(is_linear_weight_id) ==
      WebGPUGraph::ValueType::Int) {
    is_linear = graph.get_int(is_linear_weight_id) != 0;
  } else {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: is_linear_weight must be Bool or Int");
  }
  if (is_linear) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: is_linear_weight=true is unsupported");
  }

  if (weight.dims.size() < 2 || scales.dims.size() < 2 || out.dims.empty() ||
      indices.dims.empty()) {
    throw std::runtime_error("WebGPU embedding_q4gsw: malformed dims");
  }

  const uint32_t embed_dim = static_cast<uint32_t>(out.dims.back());
  if (embed_dim == 0 || embed_dim % 32 != 0) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: embed_dim must be a nonzero multiple of 32");
  }
  if (static_cast<uint64_t>(weight.dims[1]) * 2 != embed_dim) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: weight row stride mismatch (embed_dim/2)");
  }

  int64_t group_size = 0;
  if (graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int) {
    group_size = graph.get_int(group_size_id);
  }
  if (group_size <= 0) {
    throw std::runtime_error("WebGPU embedding_q4gsw: group_size <= 0");
  }

  // Leading index dims flatten row-major (mirrors Vulkan num_indices).
  const uint64_t out_numel = utils::numel_of(out.dims);
  const uint32_t num_indices = static_cast<uint32_t>(out_numel / embed_dim);
  const uint32_t groups_per_row = static_cast<uint32_t>(scales.dims[1]);
  const uint32_t blocks_per_row = embed_dim / 32u;
  const uint32_t bytes_per_row = embed_dim / 2u;
  const uint64_t total_blocks =
      static_cast<uint64_t>(num_indices) * blocks_per_row;
  if (static_cast<uint64_t>(groups_per_row) * group_size != embed_dim) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: groups_per_row * group_size != embed_dim");
  }
  if (weight.buffer == nullptr || scales.buffer == nullptr ||
      indices.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("WebGPU embedding_q4gsw: null buffer binding");
  }

  // Per-type byte guards (no runtime dtype): indices i32, weight u8, fp32 rest.
  const uint64_t indices_numel = utils::numel_of(indices.dims);
  const uint64_t weight_numel = utils::numel_of(weight.dims);
  const uint64_t scales_numel = utils::numel_of(scales.dims);
  if (indices_numel != num_indices ||
      indices.nbytes != indices_numel * sizeof(int32_t) ||
      weight.nbytes != weight_numel ||
      scales.nbytes != scales_numel * sizeof(float) ||
      out.nbytes != out_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: dtype/byte-size mismatch "
        "(indices int32, weight uint8, scales/out fp32)");
  }
  if (total_blocks > UINT32_MAX) {
    throw std::runtime_error(
        "WebGPU embedding_q4gsw: total_blocks exceeds uint32 dispatch range");
  }

  // 1D dispatch: one thread per 32-dim block; validate before any alloc.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kEmbeddingQ4gswWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(total_blocks), wg_size, "embedding_q4gsw");

  EmbeddingParams params = {};
  params.embed_dim = embed_dim;
  params.blocks_per_row = blocks_per_row;
  params.num_indices = num_indices; // std140 layout only; shader derives it
  params.group_size = static_cast<uint32_t>(group_size);
  params.groups_per_row = groups_per_row;
  params.bytes_per_row = bytes_per_row;
  params.total_blocks = static_cast<uint32_t>(total_blocks);

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(EmbeddingParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(EmbeddingParams));
  std::memcpy(mapped, &params, sizeof(EmbeddingParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(EmbeddingParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kEmbeddingQ4gswWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group layout: out (rw) + indices/weight/scales (ro storage) + uniform.
  WGPUBindGroupLayoutEntry entries[5] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  for (uint32_t i = 1; i <= 3; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
  entries[4].binding = 4;
  entries[4].visibility = WGPUShaderStage_Compute;
  entries[4].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 5;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out.buffer;
  bg_entries[0].size = out.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = indices.buffer;
  bg_entries[1].size = indices.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight.buffer;
  bg_entries[2].size = weight.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = scales.buffer;
  bg_entries[3].size = scales.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = uniform_buffer;
  bg_entries[4].size = sizeof(EmbeddingParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline, bind_group, workgroup_count, "embedding_q4gsw"});

  // Dynamic shapes: recompute counts/dispatch; out = indices + [embed_dim].
  const uint32_t gs_u = static_cast<uint32_t>(group_size);
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      indices_id,
      [indices_id,
       out_id,
       embed_dim,
       blocks_per_row,
       gs_u,
       groups_per_row,
       bytes_per_row,
       wg_size,
       dispatch_idx,
       params_buf](WebGPUGraph& g) {
        const auto& id = g.cur_dims(indices_id);
        const uint64_t ni = utils::numel_of(id);
        const uint64_t total_blocks = ni * blocks_per_row;
        if (total_blocks > UINT32_MAX) {
          throw std::runtime_error(
              "WebGPU embedding_q4gsw: total_blocks exceeds uint32");
        }
        std::vector<int64_t> od = id;
        od.push_back(static_cast<int64_t>(embed_dim));
        g.set_cur_dims(out_id, od);
        EmbeddingParams p = {};
        p.embed_dim = embed_dim;
        p.blocks_per_row = blocks_per_row;
        p.num_indices = static_cast<uint32_t>(ni);
        p.group_size = gs_u;
        p.groups_per_row = groups_per_row;
        p.bytes_per_row = bytes_per_row;
        p.total_blocks = static_cast<uint32_t>(total_blocks);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            utils::compute_1d_workgroup_count(
                g.device(),
                static_cast<uint32_t>(total_blocks),
                wg_size,
                "embedding_q4gsw(resize)");
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.embedding_q4gsw.default, embedding_q4gsw_impl);
}

} // namespace executorch::backends::webgpu

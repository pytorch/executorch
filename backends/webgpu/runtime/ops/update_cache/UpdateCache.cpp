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
#include <executorch/backends/webgpu/runtime/ops/update_cache/update_cache_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct (16-byte aligned).
struct UpdateCacheParams {
  uint32_t numel;
  uint32_t dst_offset;
  uint32_t cache_numel;
  uint32_t _pad0;
};
static_assert(
    sizeof(UpdateCacheParams) == 16,
    "UpdateCacheParams must be 16 bytes");

// llama.update_cache.default args: [value, cache, input_pos, out].
void update_cache_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int value_id = args.at(0);
  const int cache_id = args.at(1);
  const int input_pos_id = args.at(2);

  WGPUDevice device = graph.device();

  const auto& value_tensor = graph.get_tensor(value_id);
  const auto& cache_tensor = graph.get_tensor(cache_id);
  if (value_tensor.dims.size() < 4 || cache_tensor.dims.size() < 4 ||
      value_tensor.nbytes == 0) {
    throw std::runtime_error("WebGPU update_cache: expects 4D value and cache");
  }

  uint64_t value_numel = 1;
  for (int64_t d : value_tensor.dims) {
    value_numel *= static_cast<uint64_t>(d);
  }
  // fp32-only shader: bail if bytes don't match an fp32 element count.
  if (value_tensor.nbytes != value_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU update_cache: fp32-only (byte-size mismatch)");
  }

  const size_t ndim = value_tensor.dims.size();
  const size_t cndim = cache_tensor.dims.size();
  // Mirror Vulkan update_cache_impl shape guards (backends/vulkan SDPA.cpp).
  if (value_tensor.dims[ndim - 4] != 1 || cache_tensor.dims[cndim - 4] != 1) {
    throw std::runtime_error("WebGPU update_cache: batch must be 1");
  }
  if (value_tensor.dims[ndim - 1] != cache_tensor.dims[cndim - 1]) {
    throw std::runtime_error("WebGPU update_cache: head_dim mismatch");
  }
  if (value_tensor.dims[ndim - 2] != cache_tensor.dims[cndim - 2]) {
    throw std::runtime_error("WebGPU update_cache: n_heads mismatch");
  }
  const uint64_t head_dim = static_cast<uint64_t>(value_tensor.dims[ndim - 1]);
  const uint64_t n_heads = static_cast<uint64_t>(value_tensor.dims[ndim - 2]);

  uint64_t cache_numel = 1;
  for (int64_t d : cache_tensor.dims) {
    cache_numel *= static_cast<uint64_t>(d);
  }

  if (graph.get_value_type(input_pos_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error(
        "WebGPU update_cache: input_pos must be Int (SymInt not yet supported)");
  }
  const int64_t input_pos = graph.get_int(input_pos_id);
  if (input_pos < 0) {
    throw std::runtime_error(
        "WebGPU update_cache: input_pos must be non-negative");
  }

  // Bound input_pos in u64 so the u32 param downcasts cannot overflow/truncate.
  const uint64_t stride = n_heads * head_dim;
  if (cache_numel > UINT32_MAX || value_numel > cache_numel ||
      static_cast<uint64_t>(input_pos) > (cache_numel - value_numel) / stride) {
    throw std::runtime_error(
        "WebGPU update_cache: input_pos writes past cache capacity");
  }
  const uint64_t dst_offset = static_cast<uint64_t>(input_pos) * stride;

  UpdateCacheParams params = {};
  params.numel = static_cast<uint32_t>(value_numel);
  params.dst_offset = static_cast<uint32_t>(dst_offset);
  params.cache_numel = static_cast<uint32_t>(cache_numel);

  // Validate dispatch against device limits before allocating GPU objects.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kUpdateCacheWorkgroupSizeX);
  const uint32_t workgroup_count_x = utils::compute_1d_workgroup_count(
      device, params.numel, wg_size, "update_cache");

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(UpdateCacheParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(UpdateCacheParams));
  std::memcpy(mapped, &params, sizeof(UpdateCacheParams));
  wgpuBufferUnmap(uniform_buffer);

  graph.add_uniform_buffer_bytes(sizeof(UpdateCacheParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kUpdateCacheWGSL, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Bind group layout: cache (rw storage) + value (ro storage) + params.
  WGPUBindGroupLayoutEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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

  WGPUBindGroupEntry bg_entries[3] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = cache_tensor.buffer;
  bg_entries[0].size = cache_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = value_tensor.buffer;
  bg_entries[1].size = value_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = uniform_buffer;
  bg_entries[2].size = sizeof(UpdateCacheParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count_x});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Drop our ref; the bind group keeps the uniform buffer alive until release.
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(update_cache.default, update_cache_impl);
}

} // namespace executorch::backends::webgpu

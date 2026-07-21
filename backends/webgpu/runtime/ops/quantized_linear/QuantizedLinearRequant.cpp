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
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_requant_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct Q4gswRequantParams {
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t num_words;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(Q4gswRequantParams) == 32, "params must be 32 bytes");

// STE re-quant + int4 pack at a frozen per-group scale (only the codes move).
void q4gsw_requant_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int latent_id = args.at(0);
  const int scales_id = args.at(1);
  const int group_size_id = args.at(2);
  const int packed_id = args.at(3);

  WGPUDevice device = graph.device();
  const auto& latent = graph.get_tensor(latent_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& packed = graph.get_tensor(packed_id);

  if (latent.dims.size() != 2 || scales.dims.size() != 2 ||
      packed.dims.size() != 2) {
    throw std::runtime_error("q4gsw_requant: bad tensor ranks");
  }
  const uint32_t N = static_cast<uint32_t>(latent.dims[0]);
  const uint32_t K = static_cast<uint32_t>(latent.dims[1]);
  const uint32_t K_packed = static_cast<uint32_t>(packed.dims[1]);
  const uint32_t num_groups = static_cast<uint32_t>(scales.dims[0]);
  const uint32_t padded_N = static_cast<uint32_t>(scales.dims[1]);
  if (N == 0 || K == 0) {
    throw std::runtime_error("q4gsw_requant: N or K == 0");
  }
  if (static_cast<uint32_t>(packed.dims[0]) != N) {
    throw std::runtime_error("q4gsw_requant: packed rows != N");
  }
  if (K_packed != (K + 1u) / 2u) {
    throw std::runtime_error("q4gsw_requant: K_packed != ceil(K/2)");
  }
  if ((static_cast<uint64_t>(N) * K_packed) % 4u != 0u) {
    throw std::runtime_error(
        "q4gsw_requant: N*K_packed must be a multiple of 4 (u32-packed)");
  }

  if (graph.get_value_type(group_size_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("q4gsw_requant: group_size must be Int");
  }
  const int64_t group_size = graph.get_int(group_size_id);
  if (group_size <= 0) {
    throw std::runtime_error("q4gsw_requant: group_size must be positive");
  }
  const uint32_t gs = static_cast<uint32_t>(group_size);
  if (num_groups < (K + gs - 1u) / gs || padded_N < N) {
    throw std::runtime_error("q4gsw_requant: scales dims too small for K/N");
  }

  if (latent.nbytes != static_cast<uint64_t>(N) * K * sizeof(float)) {
    throw std::runtime_error("q4gsw_requant: latent fp32-only");
  }
  if (scales.nbytes !=
      static_cast<uint64_t>(num_groups) * padded_N * sizeof(float)) {
    throw std::runtime_error("q4gsw_requant: scales fp32/shape mismatch");
  }
  if (packed.nbytes != static_cast<uint64_t>(N) * K_packed) {
    throw std::runtime_error("q4gsw_requant: packed byte-size mismatch");
  }

  const uint32_t num_words =
      static_cast<uint32_t>((static_cast<uint64_t>(N) * K_packed) / 4u);

  Q4gswRequantParams params = {};
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;
  params.group_size = gs;
  params.padded_N = padded_N;
  params.num_words = num_words;

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ4gswRequantWorkgroupSizeX);
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, num_words, wg_size, "q4gsw_requant");

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(params));
  graph.add_uniform_buffer_bytes(sizeof(params));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ4gswRequantWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  for (uint32_t i = 1; i <= 2; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
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

  WGPUBindGroupEntry bg_entries[4] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = packed.buffer;
  bg_entries[0].size = packed.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = latent.buffer;
  bg_entries[1].size = latent.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = scales.buffer;
  bg_entries[2].size = scales.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(params);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count, "q4gsw_requant"});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q4gsw_requant.default, q4gsw_requant_impl);
}

} // namespace executorch::backends::webgpu

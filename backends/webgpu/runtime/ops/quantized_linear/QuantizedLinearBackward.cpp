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
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_backward_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct Q4gswBackwardParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(Q4gswBackwardParams) == 32, "params must be 32 bytes");

// linear_q4gsw_backward: d_x[M,K] = d_out[M,N] @ dequant(W)[N,K].
void q4gsw_backward_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int dout_id = args.at(0);
  const int weight_id = args.at(1);
  const int scales_id = args.at(2);
  const int group_size_id = args.at(3);
  const int dx_id = args.at(4);

  WGPUDevice device = graph.device();
  const auto& dout = graph.get_tensor(dout_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& dx = graph.get_tensor(dx_id);

  if (weight.dims.size() != 2 || scales.dims.size() != 2 || dx.dims.empty() ||
      dout.dims.empty()) {
    throw std::runtime_error("q4gsw_backward: bad tensor ranks");
  }
  const uint32_t N = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t K_packed = static_cast<uint32_t>(weight.dims[1]);
  const uint32_t K = static_cast<uint32_t>(dx.dims.back());
  if (N == 0 || K == 0) {
    throw std::runtime_error("q4gsw_backward: N or K == 0");
  }
  uint64_t dx_numel = 1;
  for (int64_t d : dx.dims) {
    dx_numel *= static_cast<uint64_t>(d);
  }
  const uint32_t M = static_cast<uint32_t>(dx_numel / K);
  const uint32_t num_groups = static_cast<uint32_t>(scales.dims[0]);
  const uint32_t padded_N = static_cast<uint32_t>(scales.dims[1]);

  if (graph.get_value_type(group_size_id) != WebGPUGraph::ValueType::Int) {
    throw std::runtime_error("q4gsw_backward: group_size must be Int");
  }
  const int64_t group_size = graph.get_int(group_size_id);
  if (group_size <= 0) {
    throw std::runtime_error("q4gsw_backward: group_size must be positive");
  }
  const uint32_t gs = static_cast<uint32_t>(group_size);

  // fp32 + shape guards (mirror the forward's byte checks).
  if (dx.nbytes != dx_numel * sizeof(float)) {
    throw std::runtime_error("q4gsw_backward: d_x fp32-only");
  }
  if (dout.nbytes != static_cast<uint64_t>(M) * N * sizeof(float)) {
    throw std::runtime_error("q4gsw_backward: d_out fp32/shape mismatch");
  }
  if (scales.nbytes !=
      static_cast<uint64_t>(num_groups) * padded_N * sizeof(float)) {
    throw std::runtime_error("q4gsw_backward: scales fp32/shape mismatch");
  }
  if (weight.nbytes != static_cast<uint64_t>(N) * K_packed) {
    throw std::runtime_error("q4gsw_backward: weight byte-size mismatch");
  }
  if (K_packed != (K + 1u) / 2u) {
    throw std::runtime_error("q4gsw_backward: K_packed != ceil(K/2)");
  }
  if (num_groups < (K + gs - 1u) / gs || padded_N < N) {
    throw std::runtime_error("q4gsw_backward: scales too small");
  }

  Q4gswBackwardParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;
  params.group_size = gs;
  params.padded_N = padded_N;

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ4gswBackwardWorkgroupSizeX);
  const uint64_t tiles = static_cast<uint64_t>((M + 3u) / 4u) * ((K + 3u) / 4u);
  if (tiles > UINT32_MAX) {
    throw std::runtime_error("q4gsw_backward: tile count exceeds u32");
  }
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(tiles), wg_size, "q4gsw_backward");

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(Q4gswBackwardParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  std::memcpy(
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(Q4gswBackwardParams)),
      &params,
      sizeof(Q4gswBackwardParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(Q4gswBackwardParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ4gswBackwardWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

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
  bg_entries[0].buffer = dx.buffer;
  bg_entries[0].size = dx.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = dout.buffer;
  bg_entries[1].size = dout.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight.buffer;
  bg_entries[2].size = weight.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = scales.buffer;
  bg_entries[3].size = scales.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = uniform_buffer;
  bg_entries[4].size = sizeof(Q4gswBackwardParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count, "q4gsw_backward"});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_q4gsw_backward.default, q4gsw_backward_impl);
}

} // namespace executorch::backends::webgpu

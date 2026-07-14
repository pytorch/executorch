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
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_dw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct Q4gswDwParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t _pad;
};
static_assert(sizeof(Q4gswDwParams) == 16, "params must be 16 bytes");

// STE weight gradient d_W[N,K] = d_out^T @ x.
void q4gsw_dw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int dout_id = args.at(0);
  const int x_id = args.at(1);
  const int dw_id = args.at(2);

  WGPUDevice device = graph.device();
  const auto& dout = graph.get_tensor(dout_id);
  const auto& x = graph.get_tensor(x_id);
  const auto& dw = graph.get_tensor(dw_id);

  if (dw.dims.size() != 2 || dout.dims.empty() || x.dims.empty()) {
    throw std::runtime_error("q4gsw_dw: bad tensor ranks");
  }
  const uint32_t N = static_cast<uint32_t>(dw.dims[0]);
  const uint32_t K = static_cast<uint32_t>(dw.dims[1]);
  if (N == 0 || K == 0) {
    throw std::runtime_error("q4gsw_dw: N or K == 0");
  }

  uint64_t dout_numel = 1;
  for (int64_t d : dout.dims) {
    dout_numel *= static_cast<uint64_t>(d);
  }
  uint64_t x_numel = 1;
  for (int64_t d : x.dims) {
    x_numel *= static_cast<uint64_t>(d);
  }
  if (static_cast<uint32_t>(dout.dims.back()) != N ||
      static_cast<uint32_t>(x.dims.back()) != K) {
    throw std::runtime_error("q4gsw_dw: d_out/x last dim mismatch");
  }
  const uint32_t M = static_cast<uint32_t>(dout_numel / N);
  if (dout_numel % N != 0 || x_numel % K != 0 || x_numel / K != M) {
    throw std::runtime_error("q4gsw_dw: M mismatch across d_out/x");
  }

  // fp32-only byte-size guards (mirror the forward's byte checks).
  if (dw.nbytes != static_cast<uint64_t>(N) * K * sizeof(float) ||
      dout.nbytes != dout_numel * sizeof(float) ||
      x.nbytes != x_numel * sizeof(float)) {
    throw std::runtime_error("q4gsw_dw: fp32-only (byte-size mismatch)");
  }

  Q4gswDwParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ4gswDwWorkgroupSizeX);
  const uint64_t tiles =
      utils::div_up<uint64_t>(N, 4u) * utils::div_up<uint64_t>(K, 4u);
  if (tiles > UINT32_MAX) {
    throw std::runtime_error("q4gsw_dw: tile count exceeds u32");
  }
  const uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(tiles), wg_size, "q4gsw_dw");

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(params));
  graph.add_uniform_buffer_bytes(sizeof(params));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ4gswDwWGSL, WGPU_STRLEN};
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
  bg_entries[0].buffer = dw.buffer;
  bg_entries[0].size = dw.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = dout.buffer;
  bg_entries[1].size = dout.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = x.buffer;
  bg_entries[2].size = x.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(params);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch({pipeline, bind_group, workgroup_count, "q4gsw_dw"});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_q4gsw_dw.default, q4gsw_dw_impl);
}

} // namespace executorch::backends::webgpu

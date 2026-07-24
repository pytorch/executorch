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
#include <executorch/backends/webgpu/runtime/ops/linear_dq8ca_q4gsw/linear_dq8ca_q4gsw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Dq8caParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(Dq8caParams) == 32, "Dq8caParams must be 32 bytes");

constexpr int64_t kTileM = 4; // MUST match TM in linear_dq8ca_q4gsw.wgsl
constexpr int64_t kTileN = 4; // MUST match TN

// et_vk.linear_dq8ca_q4gsw args (mirrors Vulkan QuantizedLinear.cpp:760):
// [in, input_scale, input_zp, weight, weight_sums, weight_scales, group_size,
//  bias, out]. Dynamic per-row int8 activation quant x 4-bit-group symmetric
// weight. weight_sums (arg 4) is a perf shortcut; this v1 recomputes the sum
// inline so it is intentionally unused. Static-shape only (no resize hook yet).
void linear_dq8ca_q4gsw_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int input_scale_id = args.at(1);
  const int input_zp_id = args.at(2);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int group_size_id = args.at(6);
  const int bias_id = args.at(7);
  const int out_id = args.at(8);

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& input_scale = graph.get_tensor(input_scale_id);
  const auto& input_zp = graph.get_tensor(input_zp_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.empty() || weight.dims.size() < 2 || scales.dims.size() < 2) {
    throw std::runtime_error("linear_dq8ca_q4gsw: malformed dims");
  }
  if (in.buffer == nullptr || input_scale.buffer == nullptr ||
      input_zp.buffer == nullptr || weight.buffer == nullptr ||
      scales.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("linear_dq8ca_q4gsw: null buffer binding");
  }

  const uint32_t K = static_cast<uint32_t>(in.dims.back());
  if (K == 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: K == 0");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  if (in_numel % K != 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: input numel % K != 0");
  }
  const uint32_t M = static_cast<uint32_t>(in_numel / K);
  const uint32_t N = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t K_packed = static_cast<uint32_t>(weight.dims[1]);
  const uint32_t num_groups = static_cast<uint32_t>(scales.dims[0]);
  const uint32_t padded_N = static_cast<uint32_t>(scales.dims[1]);
  if (M == 0 || N == 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: M or N == 0");
  }
  if (K_packed != (K + 1) / 2) {
    throw std::runtime_error("linear_dq8ca_q4gsw: K_packed must be ceil(K/2)");
  }
  if ((static_cast<uint64_t>(N) * K_packed) % 4u != 0u) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: N*K_packed must be a multiple of 4 (u32-packed)");
  }

  int64_t group_size = 0;
  if (graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int) {
    group_size = graph.get_int(group_size_id);
  }
  if (group_size <= 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: group_size <= 0");
  }
  const uint32_t gs = static_cast<uint32_t>(group_size);

  // fp32-only byte guards; per-row scale (f32[M]) + zp (int8[M]).
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != static_cast<uint64_t>(M) * N * sizeof(float) ||
      scales.nbytes !=
          static_cast<uint64_t>(num_groups) * padded_N * sizeof(float) ||
      weight.nbytes != static_cast<uint64_t>(N) * K_packed) {
    throw std::runtime_error("linear_dq8ca_q4gsw: fp32/byte-size mismatch");
  }
  // Per-row activation scale (fp32[M]) + zp (int8[M], packed 4/word in-shader).
  if (input_scale.nbytes != static_cast<uint64_t>(M) * sizeof(float) ||
      !input_zp.is_int8 || input_zp.nbytes != static_cast<uint64_t>(M)) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: input scale fp32[M] / zp int8[M] required");
  }
  // int8 zp is bound word-aligned over a max(nbytes,4) buffer; M in {5,6,7,...}
  // would bind past the buffer. Mirrors the choose_qparams_affine producer guard.
  if (M > 4u && M % 4u != 0u) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: num_rows must be <=4 or a multiple of 4");
  }
  if (num_groups < (K + gs - 1u) / gs || padded_N < N) {
    throw std::runtime_error("linear_dq8ca_q4gsw: scales dims too small");
  }

  uint32_t has_bias = 0;
  WGPUBuffer bias_buffer = nullptr;
  uint64_t bias_size = 4;
  if (graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor) {
    const auto& bias = graph.get_tensor(bias_id);
    if (bias.buffer != nullptr && bias.nbytes >= N * sizeof(float)) {
      has_bias = 1;
      bias_buffer = bias.buffer;
      bias_size = bias.nbytes;
    }
  }
  if (bias_buffer == nullptr) {
    bias_buffer = graph.create_scratch_buffer(4);
  }

  Dq8caParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;
  params.group_size = gs;
  params.padded_N = padded_N;
  params.has_bias = has_bias;

  const int64_t total_tiles = utils::div_up<int64_t>(M, kTileM) *
      utils::div_up<int64_t>(N, kTileN);
  if (total_tiles > static_cast<int64_t>(UINT32_MAX)) {
    throw std::runtime_error("linear_dq8ca_q4gsw: tile count exceeds u32");
  }
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kLinearDq8caQ4gswWorkgroupSizeX);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(total_tiles), wg_size, "linear_dq8ca_q4gsw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Dq8caParams));
  graph.add_uniform_buffer_bytes(sizeof(Dq8caParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kLinearDq8caQ4gswWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // 0 out(rw), 1 in, 2 input_scale, 3 input_zp, 4 weight, 5 scales, 6 bias (ro),
  // 7 uniform.
  WGPUBindGroupLayoutEntry entries[8] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  for (uint32_t i = 1; i <= 6; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
  entries[7].binding = 7;
  entries[7].visibility = WGPUShaderStage_Compute;
  entries[7].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 8;
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

  WGPUBindGroupEntry bg[8] = {};
  bg[0].binding = 0;
  bg[0].buffer = out.buffer;
  bg[0].size = out.nbytes;
  bg[1].binding = 1;
  bg[1].buffer = in.buffer;
  bg[1].size = in.nbytes;
  bg[2].binding = 2;
  bg[2].buffer = input_scale.buffer;
  bg[2].size = input_scale.nbytes;
  bg[3].binding = 3;
  bg[3].buffer = input_zp.buffer;
  // int8 zp bound as array<u32>; round to a multiple of 4 (buffer is >=4 bytes).
  bg[3].size = ((input_zp.nbytes + 3u) / 4u) * 4u;
  bg[4].binding = 4;
  bg[4].buffer = weight.buffer;
  bg[4].size = weight.nbytes;
  bg[5].binding = 5;
  bg[5].buffer = scales.buffer;
  bg[5].size = scales.nbytes;
  bg[6].binding = 6;
  bg[6].buffer = bias_buffer;
  bg[6].size = bias_size;
  bg[7].binding = 7;
  bg[7].buffer = params_buf;
  bg[7].size = sizeof(Dq8caParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 8;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "linear_dq8ca_q4gsw",
       workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      et_vk.linear_dq8ca_q4gsw.default,
      linear_dq8ca_q4gsw_impl);
}

} // namespace executorch::backends::webgpu


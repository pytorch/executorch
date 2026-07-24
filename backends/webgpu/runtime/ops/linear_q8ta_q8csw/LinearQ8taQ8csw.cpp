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
#include <executorch/backends/webgpu/runtime/ops/linear_q8ta_q8csw/linear_q8ta_q8csw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taQ8cswParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  int32_t input_zero_point;
  float input_scale;
  uint32_t has_bias;
  uint32_t _pad[2];
};
static_assert(
    sizeof(Q8taQ8cswParams) == 32,
    "Q8taQ8cswParams must match the WGSL Params struct (32 bytes)");

// int8-act x int8-weight -> fp32 (no requant); Vulkan QuantizedLinear.cpp:699.
void linear_q8ta_q8csw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args mirror Vulkan; weight_sums (arg 4) is unused (zp subtracted inline).
  const int in_id = args.at(0);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int out_id = args.at(args.size() - 1);
  const int bias_id = args.size() >= 8 ? args.at(6) : -1;

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(scales_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error(
        "linear_q8ta_q8csw: in/weight/scales/out not tensor");
  }
  const bool has_bias = bias_id >= 0 &&
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& scales_tensor = graph.get_tensor(scales_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || weight_tensor.buffer == nullptr ||
      scales_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("linear_q8ta_q8csw: null buffer binding");
  }
  if (weight_tensor.dims.size() != 2) {
    throw std::runtime_error("linear_q8ta_q8csw: weight must be 2D [N, K]");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));

  const uint64_t N = static_cast<uint64_t>(weight_tensor.dims.at(0));
  const uint64_t K = static_cast<uint64_t>(weight_tensor.dims.at(1));
  if (K == 0 || in_tensor.dims.empty() ||
      static_cast<uint64_t>(in_tensor.dims.back()) != K) {
    throw std::runtime_error("linear_q8ta_q8csw: input last dim must equal K");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in_tensor.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint64_t M = in_numel / K;
  if (M == 0 || N == 0 || static_cast<uint64_t>(out_tensor.dims.back()) != N) {
    throw std::runtime_error("linear_q8ta_q8csw: bad M/N shape");
  }
  if (M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX) {
    throw std::runtime_error("linear_q8ta_q8csw: dim exceeds u32");
  }
  if (M * K > UINT32_MAX || N * K > UINT32_MAX) {
    throw std::runtime_error("linear_q8ta_q8csw: M*K or N*K exceeds u32");
  }
  // int8 x/weight as array<u32> (numel%4==0); fp32 out, N free (TN=4).
  if (!in_tensor.is_int8 || in_tensor.nbytes != M * K || (M * K) % 4 != 0 ||
      !weight_tensor.is_int8 || weight_tensor.nbytes != N * K ||
      (N * K) % 4 != 0) {
    throw std::runtime_error("linear_q8ta_q8csw: int8 x/weight size mismatch");
  }
  if (out_tensor.is_int || out_tensor.nbytes != M * N * sizeof(float)) {
    throw std::runtime_error("linear_q8ta_q8csw: output must be fp32 [M, N]");
  }
  if (scales_tensor.nbytes != N * sizeof(float)) {
    throw std::runtime_error(
        "linear_q8ta_q8csw: weight_scales must be fp32 [N]");
  }
  if (has_bias) {
    const auto& b = graph.get_tensor(bias_id);
    if (b.buffer == nullptr || b.nbytes != N * sizeof(float)) {
      throw std::runtime_error("linear_q8ta_q8csw: bias must be fp32 [N]");
    }
  }

  Q8taQ8cswParams params = {};
  params.M = static_cast<uint32_t>(M);
  params.N = static_cast<uint32_t>(N);
  params.K = static_cast<uint32_t>(K);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.input_scale = static_cast<float>(input_scale);
  params.has_bias = has_bias ? 1u : 0u;

  const uint64_t n_tiles_u64 = ((M + 3) / 4) * ((N + 3) / 4);
  if (n_tiles_u64 > UINT32_MAX) {
    throw std::runtime_error(
        "linear_q8ta_q8csw: dispatch tile count exceeds u32");
  }
  const uint32_t n_tiles = static_cast<uint32_t>(n_tiles_u64);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kLinearQ8taQ8cswWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, n_tiles, wg_size, "linear_q8ta_q8csw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taQ8cswParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taQ8cswParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kLinearQ8taQ8cswWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[6] = {};
  for (int i = 0; i < 6; i++) {
    entries[i].binding = static_cast<uint32_t>(i);
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = (i == 0) ? WGPUBufferBindingType_Storage
        : (i == 5)                    ? WGPUBufferBindingType_Uniform
                                      : WGPUBufferBindingType_ReadOnlyStorage;
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

  // No-bias: bind scales as an unread placeholder (has_bias gates the read).
  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : scales_tensor.buffer;
  const uint64_t bias_size =
      has_bias ? graph.get_tensor(bias_id).nbytes : scales_tensor.nbytes;

  WGPUBindGroupEntry bg[6] = {};
  bg[0].binding = 0;
  bg[0].buffer = out_tensor.buffer;
  bg[0].size = out_tensor.nbytes;
  bg[1].binding = 1;
  bg[1].buffer = in_tensor.buffer;
  bg[1].size = in_tensor.nbytes;
  bg[2].binding = 2;
  bg[2].buffer = weight_tensor.buffer;
  bg[2].size = weight_tensor.nbytes;
  bg[3].binding = 3;
  bg[3].buffer = scales_tensor.buffer;
  bg[3].size = scales_tensor.nbytes;
  bg[4].binding = 4;
  bg[4].buffer = bias_buf;
  bg[4].size = bias_size;
  bg[5].binding = 5;
  bg[5].buffer = params_buf;
  bg[5].size = sizeof(Q8taQ8cswParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "linear_q8ta_q8csw",
       workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw_impl);
}

} // namespace executorch::backends::webgpu

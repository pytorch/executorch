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
#include <executorch/backends/webgpu/runtime/ops/q8ta_conv2d_pw/q8ta_conv2d_pw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taConvPwParams {
  uint32_t N;
  uint32_t OC;
  uint32_t IC;
  uint32_t H;
  uint32_t W;
  int32_t input_zero_point;
  int32_t output_zero_point;
  float input_scale;
  float inv_output_scale;
  uint32_t has_bias;
  uint32_t pad0;
  uint32_t pad1;
};
static_assert(
    sizeof(Q8taConvPwParams) == 48,
    "Q8taConvPwParams must match the WGSL Params struct (48 bytes)");

bool is_unit_pair(const std::vector<int64_t>& v, int64_t a, int64_t b) {
  return v.size() == 2 && v[0] == a && v[1] == b;
}

// int8 1x1 conv; per-position channel dot; mirrors Vulkan q8ta_conv2d_pw.
void q8ta_conv2d_pw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int bias_id = args.at(8);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(scales_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_conv2d_pw: in/weight/scales/out not tensor");
  }
  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  // Pointwise-standard config only; anything else is fail-loud.
  if (!is_unit_pair(graph.get_int_list(args.at(10)), 1, 1) ||
      !is_unit_pair(graph.get_int_list(args.at(11)), 0, 0) ||
      !is_unit_pair(graph.get_int_list(args.at(12)), 1, 1) ||
      graph.get_int(args.at(13)) != 1) {
    throw std::runtime_error("q8ta_conv2d_pw: only stride1/pad0/groups1");
  }
  const int act_id = args.at(args.size() - 2);
  if (graph.get_value_type(act_id) != WebGPUGraph::ValueType::String ||
      graph.get_string(act_id) != "none") {
    throw std::runtime_error(
        "q8ta_conv2d_pw: only activation='none' supported");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& scales_tensor = graph.get_tensor(scales_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || weight_tensor.buffer == nullptr ||
      scales_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_conv2d_pw: null buffer binding");
  }
  if (in_tensor.dims.size() != 4 || out_tensor.dims.size() != 4 ||
      weight_tensor.dims.size() != 2) {
    throw std::runtime_error("q8ta_conv2d_pw: in/out must be 4D, weight 2D");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));
  const double output_scale = graph.get_double(args.at(6));
  const int output_zero_point = graph.get_int(args.at(7));

  const uint64_t N = static_cast<uint64_t>(in_tensor.dims.at(0));
  const uint64_t IC = static_cast<uint64_t>(in_tensor.dims.at(1));
  const uint64_t H = static_cast<uint64_t>(in_tensor.dims.at(2));
  const uint64_t W = static_cast<uint64_t>(in_tensor.dims.at(3));
  const uint64_t OC = static_cast<uint64_t>(weight_tensor.dims.at(0));
  if (static_cast<uint64_t>(weight_tensor.dims.at(1)) != IC) {
    throw std::runtime_error("q8ta_conv2d_pw: weight must be [OC, IC]");
  }
  const uint64_t out_numel = N * OC * H * W;
  if (out_numel == 0) {
    throw std::runtime_error("q8ta_conv2d_pw: output is empty");
  }
  if (W % 4 != 0) {
    throw std::runtime_error("q8ta_conv2d_pw: W must be a multiple of 4");
  }
  if (N * OC * H * W > UINT32_MAX || N * IC * H * W > UINT32_MAX) {
    throw std::runtime_error("q8ta_conv2d_pw: numel exceeds u32");
  }
  if (!in_tensor.is_int8 || in_tensor.nbytes != N * IC * H * W ||
      (N * IC * H * W) % 4 != 0 || !weight_tensor.is_int8 ||
      weight_tensor.nbytes != OC * IC || (OC * IC) % 4 != 0 ||
      !out_tensor.is_int8 || out_tensor.nbytes != out_numel) {
    throw std::runtime_error(
        "q8ta_conv2d_pw: int8 in/weight/out size mismatch");
  }
  if (scales_tensor.nbytes != OC * sizeof(float)) {
    throw std::runtime_error("q8ta_conv2d_pw: weight_scales must be fp32 [OC]");
  }
  if (has_bias) {
    const auto& b = graph.get_tensor(bias_id);
    if (b.buffer == nullptr || b.nbytes != OC * sizeof(float)) {
      throw std::runtime_error("q8ta_conv2d_pw: bias must be fp32 [OC]");
    }
  }

  Q8taConvPwParams params = {};
  params.N = static_cast<uint32_t>(N);
  params.OC = static_cast<uint32_t>(OC);
  params.IC = static_cast<uint32_t>(IC);
  params.H = static_cast<uint32_t>(H);
  params.W = static_cast<uint32_t>(W);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);
  params.input_scale = static_cast<float>(input_scale);
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_output_scale = static_cast<float>(1.0 / output_scale);
  params.has_bias = has_bias ? 1u : 0u;

  const uint32_t num_words = static_cast<uint32_t>(out_numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taConv2dPwWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_words, wg_size, "q8ta_conv2d_pw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taConvPwParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taConvPwParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ8taConv2dPwWGSL, WGPU_STRLEN};
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
  bg[5].size = sizeof(Q8taConvPwParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "q8ta_conv2d_pw",
       workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_conv2d_pw.default, q8ta_conv2d_pw_impl);
}

} // namespace executorch::backends::webgpu

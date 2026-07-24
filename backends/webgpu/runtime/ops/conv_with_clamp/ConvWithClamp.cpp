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
#include <executorch/backends/webgpu/runtime/ops/conv_with_clamp/conv_with_clamp_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct ConvWithClampParams {
  uint32_t N;
  uint32_t IC;
  uint32_t H_in;
  uint32_t W_in;
  uint32_t OC;
  uint32_t H_out;
  uint32_t W_out;
  uint32_t Kh;
  uint32_t Kw;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t pad_h;
  uint32_t pad_w;
  uint32_t dil_h;
  uint32_t dil_w;
  uint32_t has_bias;
  uint32_t numel;
  uint32_t groups;
  uint32_t ic_per_group;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
  float output_min;
  float output_max;
};
static_assert(
    sizeof(ConvWithClampParams) == 96,
    "ConvWithClampParams must match the WGSL Params struct (96 bytes)");

std::pair<int64_t, int64_t> pair_or_throw(
    const std::vector<int64_t>& v,
    const char* msg) {
  if (v.size() != 2) {
    throw std::runtime_error(msg);
  }
  return {v.at(0), v.at(1)};
}

// A Scalar? clamp bound: Double/Int -> value; Null (absent) -> the default.
float scalar_or(WebGPUGraph& graph, int id, float dflt) {
  const auto t = graph.get_value_type(id);
  if (t == WebGPUGraph::ValueType::Double) {
    return static_cast<float>(graph.get_double(id));
  }
  if (t == WebGPUGraph::ValueType::Int) {
    return static_cast<float>(graph.get_int(id));
  }
  if (t == WebGPUGraph::ValueType::Null) {
    return dflt;
  }
  throw std::runtime_error("conv_with_clamp: unexpected clamp bound type");
}

// fp32 general conv2d (groups==1) + clamp; mirrors Vulkan conv_with_clamp.
void conv_with_clamp_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args mirror Vulkan convolution + clamp bounds; out=args.back().
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("conv_with_clamp: in/weight/out not tensor");
  }
  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || weight_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("conv_with_clamp: null buffer binding");
  }
  if (in_tensor.dims.size() != 4 || out_tensor.dims.size() != 4 ||
      weight_tensor.dims.size() != 4) {
    throw std::runtime_error("conv_with_clamp: in/out/weight must be 4D");
  }
  if (in_tensor.is_int || weight_tensor.is_int || out_tensor.is_int) {
    throw std::runtime_error("conv_with_clamp: fp32 only");
  }

  const auto [stride_h, stride_w] =
      pair_or_throw(graph.get_int_list(args.at(3)), "conv_with_clamp: stride");
  const auto [pad_h, pad_w] =
      pair_or_throw(graph.get_int_list(args.at(4)), "conv_with_clamp: padding");
  const auto [dil_h, dil_w] = pair_or_throw(
      graph.get_int_list(args.at(5)), "conv_with_clamp: dilation");
  if (graph.get_value_type(args.at(6)) != WebGPUGraph::ValueType::Bool) {
    throw std::runtime_error("conv_with_clamp: transposed must be bool");
  }
  if (graph.get_bool(args.at(6))) {
    throw std::runtime_error("conv_with_clamp: transposed unsupported");
  }
  const int64_t groups = graph.get_int(args.at(8));
  if (groups < 1) {
    throw std::runtime_error("conv_with_clamp: groups must be >= 1");
  }
  const float kInf = std::numeric_limits<float>::infinity();
  const float output_min = scalar_or(graph, args.at(9), -kInf);
  const float output_max = scalar_or(graph, args.at(10), kInf);

  const uint64_t N = static_cast<uint64_t>(in_tensor.dims.at(0));
  const uint64_t IC = static_cast<uint64_t>(in_tensor.dims.at(1));
  const uint64_t H_in = static_cast<uint64_t>(in_tensor.dims.at(2));
  const uint64_t W_in = static_cast<uint64_t>(in_tensor.dims.at(3));
  const uint64_t OC = static_cast<uint64_t>(weight_tensor.dims.at(0));
  const uint64_t Kh = static_cast<uint64_t>(weight_tensor.dims.at(2));
  const uint64_t Kw = static_cast<uint64_t>(weight_tensor.dims.at(3));
  const uint64_t H_out = static_cast<uint64_t>(out_tensor.dims.at(2));
  const uint64_t W_out = static_cast<uint64_t>(out_tensor.dims.at(3));
  // Declared output spatial dims must match the conv2d formula result.
  if (stride_h <= 0 || stride_w <= 0) {
    throw std::runtime_error("conv_with_clamp: stride must be positive");
  }
  const int64_t h_eff = static_cast<int64_t>(H_in) + 2 * pad_h -
      dil_h * (static_cast<int64_t>(Kh) - 1) - 1;
  const int64_t w_eff = static_cast<int64_t>(W_in) + 2 * pad_w -
      dil_w * (static_cast<int64_t>(Kw) - 1) - 1;
  if (h_eff < 0 || w_eff < 0 ||
      static_cast<uint64_t>(h_eff / stride_h + 1) != H_out ||
      static_cast<uint64_t>(w_eff / stride_w + 1) != W_out) {
    throw std::runtime_error(
        "conv_with_clamp: output dims inconsistent with conv2d formula");
  }
  const uint64_t numel = N * OC * H_out * W_out;
  const uint64_t ug = static_cast<uint64_t>(groups);
  // Grouped weight is [OC, IC/groups, Kh, Kw]; ic_per_group = weight.dims[1].
  const uint64_t ic_per_group =
      static_cast<uint64_t>(weight_tensor.dims.at(1));
  if (IC == 0 || numel == 0 || numel > UINT32_MAX || IC % ug != 0 ||
      OC % ug != 0 || ic_per_group * ug != IC) {
    throw std::runtime_error("conv_with_clamp: bad shape (IC/numel/groups)");
  }
  if (out_tensor.nbytes != numel * sizeof(float) ||
      in_tensor.nbytes != N * IC * H_in * W_in * sizeof(float) ||
      weight_tensor.nbytes != OC * ic_per_group * Kh * Kw * sizeof(float)) {
    throw std::runtime_error("conv_with_clamp: fp32 byte-size mismatch");
  }
  if (has_bias) {
    const auto& b = graph.get_tensor(bias_id);
    if (b.buffer == nullptr || b.nbytes != OC * sizeof(float)) {
      throw std::runtime_error("conv_with_clamp: bias must be fp32 [OC]");
    }
  }

  ConvWithClampParams params = {};
  params.N = static_cast<uint32_t>(N);
  params.IC = static_cast<uint32_t>(IC);
  params.H_in = static_cast<uint32_t>(H_in);
  params.W_in = static_cast<uint32_t>(W_in);
  params.OC = static_cast<uint32_t>(OC);
  params.H_out = static_cast<uint32_t>(H_out);
  params.W_out = static_cast<uint32_t>(W_out);
  params.Kh = static_cast<uint32_t>(Kh);
  params.Kw = static_cast<uint32_t>(Kw);
  params.stride_h = static_cast<uint32_t>(stride_h);
  params.stride_w = static_cast<uint32_t>(stride_w);
  params.pad_h = static_cast<uint32_t>(pad_h);
  params.pad_w = static_cast<uint32_t>(pad_w);
  params.dil_h = static_cast<uint32_t>(dil_h);
  params.dil_w = static_cast<uint32_t>(dil_w);
  params.has_bias = has_bias ? 1u : 0u;
  params.numel = static_cast<uint32_t>(numel);
  params.groups = static_cast<uint32_t>(groups);
  params.ic_per_group = static_cast<uint32_t>(ic_per_group);
  params.output_min = output_min;
  params.output_max = output_max;

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kConvWithClampWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(numel), wg_size, "conv_with_clamp");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(ConvWithClampParams));
  graph.add_uniform_buffer_bytes(sizeof(ConvWithClampParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kConvWithClampWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // out (rw) + in/weight/bias (ro storage) + params (uniform).
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

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  pipeline_desc.compute.constantCount = 1;
  pipeline_desc.compute.constants = &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  // No-bias: bind weight as an unread placeholder (has_bias gates the read).
  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : weight_tensor.buffer;
  const uint64_t bias_size =
      has_bias ? graph.get_tensor(bias_id).nbytes : weight_tensor.nbytes;

  WGPUBindGroupEntry bg[5] = {};
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
  bg[3].buffer = bias_buf;
  bg[3].size = bias_size;
  bg[4].binding = 4;
  bg[4].buffer = params_buf;
  bg[4].size = sizeof(ConvWithClampParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "conv_with_clamp",
       workgroup_count.y});

  // conv2d is static-shape-only: no tensor resize hook is registered, so the
  // output spatial dims stay fixed at their build-time (serialized) values.
  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.conv_with_clamp.default, conv_with_clamp_impl);
}

} // namespace executorch::backends::webgpu

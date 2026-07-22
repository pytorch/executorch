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
#include <executorch/backends/webgpu/runtime/ops/conv1d_dw/conv1d_dw_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/conv1d_dw/conv1d_pw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Conv1dDwParams {
  uint32_t kernel_size;
  uint32_t stride;
  uint32_t padding;
  uint32_t dilation;
  uint32_t channels;
  uint32_t in_len;
  uint32_t out_len;
  uint32_t numel;
  uint32_t has_bias;
  uint32_t pad0;
  float output_min;
  float output_max;
};
static_assert(
    sizeof(Conv1dDwParams) == 48,
    "Conv1dDwParams must match the WGSL Params struct (48 bytes)");

// Convolved output length: (L + 2p - dilation*(K-1) - 1) / stride + 1 (floor).
uint32_t conv1d_out_len(
    int64_t in_len,
    int64_t k,
    int64_t stride,
    int64_t padding,
    int64_t dilation) {
  return static_cast<uint32_t>(
      (in_len + 2 * padding - dilation * (k - 1) - 1) / stride + 1);
}

int64_t first_int(const std::vector<int64_t>& v) {
  return v.empty() ? 0 : v[0];
}

struct Conv1dPwParams {
  uint32_t in_channels;
  uint32_t out_channels;
  uint32_t length;
  uint32_t numel;
  uint32_t has_bias;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
static_assert(
    sizeof(Conv1dPwParams) == 32,
    "Conv1dPwParams must match the WGSL Params struct (32 bytes)");

// Pointwise conv1d (K=1, groups=1): a per-position matmul over channels.
void add_conv1d_pw_node(
    WebGPUGraph& graph,
    int in_id,
    int weight_id,
    int bias_id,
    int out_id) {
  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;
  const uint32_t in_channels = static_cast<uint32_t>(in_tensor.dims.at(1));
  const uint32_t out_channels = static_cast<uint32_t>(out_tensor.dims.at(1));
  const uint32_t length = static_cast<uint32_t>(out_tensor.dims.at(2));
  if (in_tensor.dims.at(2) != out_tensor.dims.at(2)) {
    throw std::runtime_error("conv1d_pw: in/out length (dim 2) mismatch");
  }

  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes != out_numel * sizeof(float) ||
      weight_tensor.nbytes !=
          static_cast<size_t>(out_channels) * in_channels * sizeof(float)) {
    throw std::runtime_error("conv1d_pw: fp32-only (byte-size mismatch)");
  }
  if (out_numel > UINT32_MAX) {
    throw std::runtime_error("conv1d_pw: output numel exceeds u32");
  }

  Conv1dPwParams params = {};
  params.in_channels = in_channels;
  params.out_channels = out_channels;
  params.length = length;
  params.numel = static_cast<uint32_t>(out_numel);
  params.has_bias = has_bias ? 1u : 0u;

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kConv1dPwWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "conv1d_pw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Conv1dPwParams));
  graph.add_uniform_buffer_bytes(sizeof(Conv1dPwParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kConv1dPwWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[5] = {};
  for (int i = 0; i < 4; i++) {
    entries[i].binding = static_cast<uint32_t>(i);
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = (i == 1) ? WGPUBufferBindingType_Storage
                                      : WGPUBufferBindingType_ReadOnlyStorage;
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

  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : weight_tensor.buffer;
  uint64_t bias_sz =
      has_bias ? graph.get_tensor(bias_id).nbytes : weight_tensor.nbytes;

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight_tensor.buffer;
  bg_entries[2].size = weight_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = bias_buf;
  bg_entries[3].size = bias_sz;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = params_buf;
  bg_entries[4].size = sizeof(Conv1dPwParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "conv1d_pw",
       workgroup_count.y});

  // Dynamic shapes: only the length varies; recompute params + dispatch.
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       in_channels,
       out_channels,
       has_bias,
       wg_size,
       dispatch_idx,
       p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() != 3) {
          throw std::runtime_error("conv1d_pw(resize): input is not 3D");
        }
        if (d[1] != static_cast<int64_t>(in_channels)) {
          throw std::runtime_error(
              "conv1d_pw(resize): in channel count changed");
        }
        Conv1dPwParams p = {};
        p.in_channels = static_cast<uint32_t>(d[1]);
        p.out_channels = out_channels;
        p.length = static_cast<uint32_t>(d[2]);
        p.numel = static_cast<uint32_t>(
            static_cast<uint64_t>(d[0]) * out_channels * d[2]);
        p.has_bias = has_bias ? 1u : 0u;
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "conv1d_pw(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        const std::vector<int64_t> out_d = {d[0], out_channels, d[2]};
        g.set_cur_dims(out_id, out_d);
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

// depthwise-conv1d (groups==C); mirrors Vulkan conv1d_dw (Convolution.cpp:755).
void convolution_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args mirror Vulkan conv1d_dw; bias (arg 2) may be Null; out=args.back().
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int stride_id = args.at(3);
  const int padding_id = args.at(4);
  const int dilation_id = args.at(5);
  const int transposed_id = args.at(6);
  const int groups_id = args.at(8);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("convolution: in/weight/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.dims.size() != 3 || out_tensor.dims.size() != 3 ||
      weight_tensor.dims.size() != 3) {
    throw std::runtime_error("convolution: only conv1d (3D) is supported");
  }

  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  const uint32_t channels = static_cast<uint32_t>(in_tensor.dims.at(1));
  const uint32_t in_len = static_cast<uint32_t>(in_tensor.dims.at(2));
  const uint32_t out_len = static_cast<uint32_t>(out_tensor.dims.at(2));
  const uint32_t kernel_size = static_cast<uint32_t>(weight_tensor.dims.at(2));

  const bool transposed = graph.get_bool(transposed_id);
  const int64_t groups = graph.get_int(groups_id);

  // Pointwise (K=1, groups=1): a matmul over channels; stride-1 / no-pad only.
  if (!transposed && groups == 1 && weight_tensor.dims.at(2) == 1 &&
      first_int(graph.get_int_list(stride_id)) == 1 &&
      first_int(graph.get_int_list(padding_id)) == 0) {
    add_conv1d_pw_node(graph, in_id, weight_id, bias_id, out_id);
    return;
  }

  // Otherwise only the depthwise config (groups==C, weight [C,1,K]).
  if (transposed || groups != static_cast<int64_t>(channels) ||
      weight_tensor.dims.at(0) != static_cast<int64_t>(channels) ||
      weight_tensor.dims.at(1) != 1) {
    throw std::runtime_error(
        "convolution: only depthwise or pointwise conv1d supported");
  }

  const int64_t stride_i = first_int(graph.get_int_list(stride_id));
  const int64_t padding_i = first_int(graph.get_int_list(padding_id));
  const int64_t dilation_i = first_int(graph.get_int_list(dilation_id));
  if (stride_i < 1) {
    throw std::runtime_error("convolution: stride must be >= 1");
  }
  if (padding_i < 0) {
    throw std::runtime_error("convolution: padding must be >= 0");
  }
  if (dilation_i < 1) {
    throw std::runtime_error("convolution: dilation must be >= 1");
  }
  const uint32_t stride = static_cast<uint32_t>(stride_i);
  const uint32_t padding = static_cast<uint32_t>(padding_i);
  const uint32_t dilation = static_cast<uint32_t>(dilation_i);

  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes != out_numel * sizeof(float) ||
      weight_tensor.nbytes !=
          static_cast<size_t>(channels) * kernel_size * sizeof(float)) {
    throw std::runtime_error("convolution: fp32-only (byte-size mismatch)");
  }
  if (out_numel > UINT32_MAX) {
    throw std::runtime_error("convolution: output numel exceeds u32");
  }

  // aten.convolution has no fused clamp (that is et_vk.conv_with_clamp).
  const float output_min = std::numeric_limits<float>::lowest();
  const float output_max = std::numeric_limits<float>::max();

  Conv1dDwParams params = {};
  params.kernel_size = kernel_size;
  params.stride = stride;
  params.padding = padding;
  params.dilation = dilation;
  params.channels = channels;
  params.in_len = in_len;
  params.out_len = out_len;
  params.numel = static_cast<uint32_t>(out_numel);
  params.has_bias = has_bias ? 1u : 0u;
  params.output_min = output_min;
  params.output_max = output_max;

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kConv1dDwWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "conv1d_dw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Conv1dDwParams));
  graph.add_uniform_buffer_bytes(sizeof(Conv1dDwParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kConv1dDwWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[5] = {};
  for (int i = 0; i < 4; i++) {
    entries[i].binding = static_cast<uint32_t>(i);
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = (i == 1) ? WGPUBufferBindingType_Storage
                                      : WGPUBufferBindingType_ReadOnlyStorage;
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

  // No-bias binds the weight buffer as an unread placeholder (has_bias gates).
  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : weight_tensor.buffer;
  uint64_t bias_sz =
      has_bias ? graph.get_tensor(bias_id).nbytes : weight_tensor.nbytes;

  WGPUBindGroupEntry bg_entries[5] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight_tensor.buffer;
  bg_entries[2].size = weight_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = bias_buf;
  bg_entries[3].size = bias_sz;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = params_buf;
  bg_entries[4].size = sizeof(Conv1dDwParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 5;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "conv1d_dw",
       workgroup_count.y});

  // Dynamic shapes: recompute out_len + params + dispatch.
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       channels,
       kernel_size,
       stride,
       padding,
       dilation,
       has_bias,
       output_min,
       output_max,
       wg_size,
       dispatch_idx,
       p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() != 3) {
          throw std::runtime_error("conv1d_dw(resize): input is not 3D");
        }
        if (d[1] != static_cast<int64_t>(channels)) {
          throw std::runtime_error("conv1d_dw(resize): channel count changed");
        }
        Conv1dDwParams p = {};
        p.kernel_size = kernel_size;
        p.stride = stride;
        p.padding = padding;
        p.dilation = dilation;
        p.channels = static_cast<uint32_t>(d[1]);
        p.in_len = static_cast<uint32_t>(d[2]);
        p.out_len =
            conv1d_out_len(d[2], kernel_size, stride, padding, dilation);
        const uint64_t out_numel =
            static_cast<uint64_t>(d[0]) * d[1] * p.out_len;
        if (out_numel > UINT32_MAX) {
          throw std::runtime_error(
              "conv1d_dw(resize): output numel exceeds u32");
        }
        p.numel = static_cast<uint32_t>(out_numel);
        p.has_bias = has_bias ? 1u : 0u;
        p.output_min = output_min;
        p.output_max = output_max;
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "conv1d_dw(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        const std::vector<int64_t> out_d = {
            d[0], d[1], static_cast<int64_t>(p.out_len)};
        g.set_cur_dims(out_id, out_d);
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.convolution.default, convolution_impl);
}

} // namespace executorch::backends::webgpu

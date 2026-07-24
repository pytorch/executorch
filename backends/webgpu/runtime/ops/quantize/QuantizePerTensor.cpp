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
#include <executorch/backends/webgpu/runtime/ops/quantize/quantize_per_tensor_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct QuantParams {
  float inv_scale;
  int32_t zero_point;
  uint32_t numel;
  uint32_t pad0;
};
static_assert(
    sizeof(QuantParams) == 16,
    "QuantParams must match the WGSL Params struct (16 bytes)");

// fp32->int8; mirrors Vulkan q8ta_quantize.glsl (round(x * inv_scale) + zp).
void quantize_per_tensor_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  // args: [x, scale, zp, qmin, qmax, dtype, out]; out is always args.back().
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("quantize_per_tensor: in/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("quantize_per_tensor: null buffer binding");
  }

  const double scale = graph.get_double(args.at(1));
  const int zero_point = graph.get_int(args.at(2));

  uint64_t numel = 1;
  for (int64_t d : in_tensor.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  // int8 buffer is bound as array<u32>; require a whole number of 4-elem words.
  if (numel == 0 || numel % 4 != 0) {
    throw std::runtime_error(
        "quantize_per_tensor: numel must be a nonzero "
        "multiple of 4");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("quantize_per_tensor: numel exceeds u32");
  }
  if (in_tensor.nbytes != numel * sizeof(float)) {
    throw std::runtime_error("quantize_per_tensor: input is not fp32");
  }
  // int8 output (not uint8/bool): the kernel hardcodes the [-128,127] clamp.
  if (!out_tensor.is_int8 || out_tensor.nbytes != numel) {
    throw std::runtime_error("quantize_per_tensor: output is not int8");
  }

  QuantParams params = {};
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_scale = static_cast<float>(1.0 / scale);
  params.zero_point = static_cast<int32_t>(zero_point);
  params.numel = static_cast<uint32_t>(numel);

  const uint32_t num_words = static_cast<uint32_t>(numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQuantizePerTensorWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_words, wg_size, "quantize_per_tensor");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(QuantParams));
  graph.add_uniform_buffer_bytes(sizeof(QuantParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQuantizePerTensorWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[3] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
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
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = params_buf;
  bg_entries[2].size = sizeof(QuantParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "quantize_per_tensor",
       workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      quantized_decomposed.quantize_per_tensor.default,
      quantize_per_tensor_impl);
}

} // namespace executorch::backends::webgpu

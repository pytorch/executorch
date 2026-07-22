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
#include <executorch/backends/webgpu/runtime/ops/to_copy/to_copy_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct ToCopyParams {
  uint32_t numel;
  uint32_t convert_mode; // 0 = raw copy, 1 = int->float, 2 = float->int
  uint32_t pad0;
  uint32_t pad1;
};
static_assert(sizeof(ToCopyParams) == 16, "ToCopyParams must be 16 bytes");

// _to_copy / _to_dim_order_copy: same-dtype = raw copy; int<->float must CONVERT
// (the dtype-promotion pass inserts int->float _to_copy before binary ops).
void to_copy_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("to_copy: null buffer binding");
  }
  // 4-byte elements only: numel below and the int<->float convert path assume a
  // 4-byte element. to_copy legitimately handles fp32 AND int32, so guard on
  // element size (not fp32-only) and reject bool/fp16/int64.
  if (in_tensor.elem_size != sizeof(float) ||
      out_tensor.elem_size != sizeof(float)) {
    throw std::runtime_error(
        "to_copy: only 4-byte element types (fp32/int32) are supported");
  }
  const uint64_t numel = out_tensor.nbytes / sizeof(float);
  if (numel == 0 || numel > UINT32_MAX) {
    throw std::runtime_error("to_copy: output numel is zero or exceeds u32");
  }
  if (in_tensor.nbytes != out_tensor.nbytes) {
    throw std::runtime_error("to_copy: input/output size mismatch");
  }
  // int<->float differ only in interpretation of the same 4 bytes (nbytes equal),
  // so a raw copy would ship the wrong bit pattern; select a converting shader.
  uint32_t convert_mode = 0;
  if (in_tensor.is_int && !out_tensor.is_int) {
    convert_mode = 1; // int -> float
  } else if (!in_tensor.is_int && out_tensor.is_int) {
    convert_mode = 2; // float -> int
  }

  uint32_t wg_size = utils::clamp_workgroup_size(device, kToCopyWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(numel), wg_size, "to_copy");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  ToCopyParams params = {};
  params.numel = static_cast<uint32_t>(numel);
  params.convert_mode = convert_mode;
  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(ToCopyParams));
  graph.add_uniform_buffer_bytes(sizeof(ToCopyParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kToCopyWGSL, WGPU_STRLEN};
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
  bg_entries[2].size = sizeof(ToCopyParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  graph.add_dispatch(
      {pipeline, bind_group, workgroup_count.x, "to_copy", workgroup_count.y});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten._to_copy.default, to_copy_impl);
  WEBGPU_REGISTER_OP(dim_order_ops._to_dim_order_copy.default, to_copy_impl);
}

} // namespace executorch::backends::webgpu

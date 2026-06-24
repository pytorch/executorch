/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_vec4_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned).
struct RmsNormParams {
  uint32_t num_rows;
  uint32_t row_width;
  float epsilon;
  uint32_t _pad;
};
static_assert(sizeof(RmsNormParams) == 16, "RmsNormParams must be 16 bytes");

void rms_norm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // et_vk.rms_norm.default args: [in, weight, eps, out]
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int eps_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  // Get epsilon (Double from a Python float; defaults to float32 eps)
  float epsilon = std::numeric_limits<float>::epsilon();
  if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Double) {
    epsilon = static_cast<float>(graph.get_double(eps_id));
  } else if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Int) {
    epsilon = static_cast<float>(graph.get_int(eps_id));
  }

  // row_width = last dim; num_rows = product of the rest (PyTorch NCHW order)
  const auto& in_tensor = graph.get_tensor(in_id);
  if (in_tensor.dims.empty() || in_tensor.nbytes == 0) {
    throw std::runtime_error("WebGPU rms_norm: empty input");
  }
  const uint32_t row_width = static_cast<uint32_t>(in_tensor.dims.back());
  if (row_width == 0) {
    throw std::runtime_error("WebGPU rms_norm: zero row width");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in_tensor.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  // fp32-only shader: bail if the bytes don't match an fp32 element count.
  if (in_tensor.nbytes != in_numel * sizeof(float)) {
    throw std::runtime_error("WebGPU rms_norm: fp32-only (byte-size mismatch)");
  }
  const uint32_t num_rows = static_cast<uint32_t>(in_numel / row_width);
  if (num_rows == 0) {
    throw std::runtime_error("WebGPU rms_norm: zero rows");
  }
  // Validate the 1D dispatch limit before allocating any GPU objects.
  if (num_rows > 65535u) {
    throw std::runtime_error(
        "WebGPU rms_norm: num_rows exceeds the 1D dispatch limit (65535)");
  }

  // Create uniform buffer for params
  RmsNormParams params = {};
  params.num_rows = num_rows;
  params.row_width = row_width;
  params.epsilon = epsilon;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(RmsNormParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(RmsNormParams));
  std::memcpy(mapped, &params, sizeof(RmsNormParams));
  wgpuBufferUnmap(uniform_buffer);

  graph.add_uniform_buffer_bytes(sizeof(RmsNormParams));

  // Select the vec4 kernel when the row width is a multiple of 4 (every Llama
  // hidden size qualifies); fall back to the scalar kernel otherwise. The two
  // kernels are equivalent up to floating-point reassociation (the vec4
  // reduction reorders the sum, so not bit-identical) and share the same bind
  // group + dispatch.
  const bool use_vec4 = (row_width % 4u == 0u);

  // Create shader module from built-in WGSL source
  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {use_vec4 ? kRmsNormVec4WGSL : kRmsNormWGSL, WGPU_STRLEN};

  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  // Create bind group layout: out (rw) + in/weight (ro storage) + params
  WGPUBindGroupLayoutEntry entries[4] = {};

  // t_out - storage buffer, read-write
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;

  // t_in - storage buffer, read-only
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  // t_weight - storage buffer, read-only
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

  // params - uniform buffer
  entries[3].binding = 3;
  entries[3].visibility = WGPUShaderStage_Compute;
  entries[3].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 4;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  // Create pipeline layout
  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  // Create compute pipeline
  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

  // Create bind group with actual buffers
  const auto& out_tensor = graph.get_tensor(out_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);

  WGPUBindGroupEntry bg_entries[4] = {};

  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out_tensor.buffer;
  bg_entries[0].size = out_tensor.nbytes;

  bg_entries[1].binding = 1;
  bg_entries[1].buffer = in_tensor.buffer;
  bg_entries[1].size = in_tensor.nbytes;

  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight_tensor.buffer;
  bg_entries[2].size = weight_tensor.nbytes;

  bg_entries[3].binding = 3;
  bg_entries[3].buffer = uniform_buffer;
  bg_entries[3].size = sizeof(RmsNormParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  // One workgroup per row (kRmsNormWorkgroupSizeX threads cooperate per row)
  static_assert(
      kRmsNormWorkgroupSizeX == 64,
      "must match @workgroup_size and WG_SIZE in rms_norm.wgsl");
  static_assert(
      kRmsNormVec4WorkgroupSizeX == 64,
      "must match @workgroup_size and WG_SIZE in rms_norm_vec4.wgsl");
  graph.add_dispatch({pipeline, bind_group, num_rows});

  // Release intermediate objects (pipeline and bind_group are kept by dispatch)
  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Drop our ref; the bind group keeps the uniform buffer alive until release.
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.rms_norm.default, rms_norm_impl);
}

} // namespace executorch::backends::webgpu

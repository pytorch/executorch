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
#include <executorch/backends/webgpu/runtime/ops/rope/apply_rotary_emb_interleaved_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct InterleavedParams {
  uint32_t seq;
  uint32_t width;
  uint32_t numel;
  uint32_t pad0;
};
static_assert(
    sizeof(InterleavedParams) == 16,
    "InterleavedParams must match the WGSL Params struct (16 bytes)");

// Pair-interleaved rope; mirrors Vulkan apply_rotary_emb_interleaved.glsl.
void apply_rotary_emb_interleaved_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  // args: [x, freqs, out].
  const int in_id = args.at(0);
  const int freqs_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(freqs_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("rope_interleaved: in/freqs/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& freqs_tensor = graph.get_tensor(freqs_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  // Require rank 3 [B,N,C]; a 4D input would mis-index freqs (mirrors Vulkan).
  if (in_tensor.dims.size() != 3 || out_tensor.dims.size() != 3) {
    throw std::runtime_error("rope_interleaved: in/out must be 3D [B, N, C]");
  }
  if (in_tensor.buffer == nullptr || freqs_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("rope_interleaved: null buffer binding");
  }

  const uint32_t width = static_cast<uint32_t>(in_tensor.dims.back());
  const uint32_t seq =
      static_cast<uint32_t>(in_tensor.dims[in_tensor.dims.size() - 2]);
  if (width == 0 || width % 2 != 0) {
    throw std::runtime_error(
        "rope_interleaved: last dim must be a multiple of 2");
  }

  uint64_t numel = 1;
  for (int64_t d : in_tensor.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  const uint64_t freqs_numel = utils::numel_of(freqs_tensor.dims);
  if (in_tensor.nbytes != numel * sizeof(float) ||
      out_tensor.nbytes != numel * sizeof(float) ||
      freqs_numel != static_cast<uint64_t>(seq) * width ||
      freqs_tensor.nbytes != freqs_numel * sizeof(float)) {
    throw std::runtime_error(
        "rope_interleaved: fp32 byte mismatch or freqs != [seq, width]");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("rope_interleaved: numel exceeds u32");
  }

  InterleavedParams params = {};
  params.seq = seq;
  params.width = width;
  params.numel = static_cast<uint32_t>(numel);

  uint32_t wg_size = utils::clamp_workgroup_size(
      device, kApplyRotaryEmbInterleavedWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(numel), wg_size, "rope_interleaved");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(InterleavedParams));
  graph.add_uniform_buffer_bytes(sizeof(InterleavedParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kApplyRotaryEmbInterleavedWGSL, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

  WGPUBindGroupLayoutEntry entries[4] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  entries[1].binding = 1;
  entries[1].visibility = WGPUShaderStage_Compute;
  entries[1].buffer.type = WGPUBufferBindingType_Storage;
  entries[2].binding = 2;
  entries[2].visibility = WGPUShaderStage_Compute;
  entries[2].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
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
  bg_entries[0].buffer = in_tensor.buffer;
  bg_entries[0].size = in_tensor.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = out_tensor.buffer;
  bg_entries[1].size = out_tensor.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = freqs_tensor.buffer;
  bg_entries[2].size = freqs_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = params_buf;
  bg_entries[3].size = sizeof(InterleavedParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "rope_interleaved",
       workgroup_count.y});

  // Dynamic shapes: recompute seq/numel + dispatch (freqs stay max-allocated).
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, width, wg_size, dispatch_idx, p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() != 3) {
          throw std::runtime_error("rope_interleaved(resize): rank must be 3");
        }
        // width is baked into the params + freqs allocation; only seq may vary.
        if (d.back() != static_cast<int64_t>(width)) {
          throw std::runtime_error(
              "rope_interleaved(resize): last dim (width) changed");
        }
        const uint64_t numel = utils::numel_of(d);
        if (numel > UINT32_MAX) {
          throw std::runtime_error(
              "rope_interleaved(resize): numel exceeds u32");
        }
        InterleavedParams p = {};
        p.seq = static_cast<uint32_t>(d[d.size() - 2]);
        p.width = width;
        p.numel = static_cast<uint32_t>(numel);
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "rope_interleaved(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        g.set_cur_dims(out_id, d);
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(
      et_vk.apply_rotary_emb_interleaved.default,
      apply_rotary_emb_interleaved_impl);
}

} // namespace executorch::backends::webgpu

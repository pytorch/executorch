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
#include <executorch/backends/webgpu/runtime/ops/grid_sampler_2d/grid_sampler_2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct GridSamplerParams {
  uint32_t in_h;
  uint32_t in_w;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t channels;
  uint32_t numel;
  uint32_t pad0;
  uint32_t pad1;
};
static_assert(
    sizeof(GridSamplerParams) == 32,
    "GridSamplerParams must match the WGSL Params struct (32 bytes)");

// grid_sampler_2d: bilinear+border+align_corners sample (Vulkan glsl).
void grid_sampler_2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [in, grid, interpolation_mode, padding_mode, align_corners, out].
  const int in_id = args.at(0);
  const int grid_id = args.at(1);
  const int interp_id = args.at(2);
  const int padding_id = args.at(3);
  const int align_id = args.at(4);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(grid_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("grid_sampler_2d: in/grid/out is not a tensor");
  }

  // Mirror Vulkan's config guards (bilinear=0, border=1, align_corners=true).
  if (graph.get_int(interp_id) != 0) {
    throw std::runtime_error("grid_sampler_2d: only bilinear is supported");
  }
  if (graph.get_int(padding_id) != 1) {
    throw std::runtime_error("grid_sampler_2d: only border padding supported");
  }
  if (!graph.get_bool(align_id)) {
    throw std::runtime_error("grid_sampler_2d: requires align_corners=true");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& grid_tensor = graph.get_tensor(grid_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.dims.size() != 4 || out_tensor.dims.size() != 4 ||
      grid_tensor.dims.size() != 4) {
    throw std::runtime_error("grid_sampler_2d: in/out/grid must be 4D");
  }

  const uint32_t channels = static_cast<uint32_t>(in_tensor.dims.at(1));
  const uint32_t in_h = static_cast<uint32_t>(in_tensor.dims.at(2));
  const uint32_t in_w = static_cast<uint32_t>(in_tensor.dims.at(3));
  const uint32_t out_h = static_cast<uint32_t>(out_tensor.dims.at(2));
  const uint32_t out_w = static_cast<uint32_t>(out_tensor.dims.at(3));
  if (in_h < 1u || in_w < 1u) {
    throw std::runtime_error("grid_sampler_2d: input H/W must be >= 1");
  }

  uint64_t in_numel = 1;
  for (int64_t d : in_tensor.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  // grid is [N, out_h, out_w, 2] fp32; validate its own shape (batch matches
  // out, trailing coord pair is 2) and derive numel from the grid's dims.
  if (grid_tensor.dims.at(0) != out_tensor.dims.at(0)) {
    throw std::runtime_error("grid_sampler_2d: grid/out batch mismatch");
  }
  if (grid_tensor.dims.at(3) != 2) {
    throw std::runtime_error("grid_sampler_2d: grid last dim must be 2");
  }
  if (static_cast<uint32_t>(grid_tensor.dims.at(1)) != out_h ||
      static_cast<uint32_t>(grid_tensor.dims.at(2)) != out_w) {
    throw std::runtime_error("grid_sampler_2d: grid H/W must equal out H/W");
  }
  uint64_t grid_numel = 1;
  for (int64_t d : grid_tensor.dims) {
    grid_numel *= static_cast<uint64_t>(d);
  }
  if (in_tensor.nbytes != in_numel * sizeof(float) ||
      out_tensor.nbytes != out_numel * sizeof(float) ||
      grid_tensor.nbytes != grid_numel * sizeof(float)) {
    throw std::runtime_error("grid_sampler_2d: fp32-only (byte-size mismatch)");
  }
  if (out_numel > UINT32_MAX) {
    throw std::runtime_error("grid_sampler_2d: output numel exceeds u32");
  }

  GridSamplerParams params = {};
  params.in_h = in_h;
  params.in_w = in_w;
  params.out_h = out_h;
  params.out_w = out_w;
  params.channels = channels;
  params.numel = static_cast<uint32_t>(out_numel);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kGridSampler2dWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "grid_sampler_2d");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(GridSamplerParams));
  graph.add_uniform_buffer_bytes(sizeof(GridSamplerParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kGridSampler2dWGSL, WGPU_STRLEN};
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
  bg_entries[2].buffer = grid_tensor.buffer;
  bg_entries[2].size = grid_tensor.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = params_buf;
  bg_entries[3].size = sizeof(GridSamplerParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 4;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "grid_sampler_2d",
       workgroup_count.y});

  // Dynamic shapes: out shape depends on grid, so trigger on BOTH in and grid.
  WGPUBuffer p_buf = params_buf;
  auto gs_resize =
      [in_id, grid_id, out_id, wg_size, dispatch_idx, p_buf](WebGPUGraph& g) {
        const auto& id = g.cur_dims(in_id);
        const auto& gd = g.cur_dims(grid_id);
        if (id.size() != 4 || gd.size() != 4) {
          throw std::runtime_error("grid_sampler_2d(resize): in/grid not 4D");
        }
        if (gd[0] != id[0]) {
          throw std::runtime_error(
              "grid_sampler_2d(resize): grid/out batch mismatch");
        }
        if (gd[3] != 2) {
          throw std::runtime_error(
              "grid_sampler_2d(resize): grid last dim must be 2");
        }
        if (id[2] < 1 || id[3] < 1) {
          throw std::runtime_error(
              "grid_sampler_2d(resize): input H/W must be >= 1");
        }
        GridSamplerParams p = {};
        p.in_h = static_cast<uint32_t>(id[2]);
        p.in_w = static_cast<uint32_t>(id[3]);
        p.out_h = static_cast<uint32_t>(gd[1]);
        p.out_w = static_cast<uint32_t>(gd[2]);
        p.channels = static_cast<uint32_t>(id[1]);
        p.numel = static_cast<uint32_t>(
            static_cast<uint64_t>(id[0]) * id[1] * p.out_h * p.out_w);
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "grid_sampler_2d(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        const std::vector<int64_t> out_d = {
            id[0],
            id[1],
            static_cast<int64_t>(p.out_h),
            static_cast<int64_t>(p.out_w)};
        g.set_cur_dims(out_id, out_d);
      };
  graph.add_tensor_resize_hook(in_id, gs_resize);
  graph.add_tensor_resize_hook(grid_id, gs_resize);

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.grid_sampler_2d.default, grid_sampler_2d_impl);
}

} // namespace executorch::backends::webgpu

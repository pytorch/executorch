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
#include <executorch/backends/webgpu/runtime/ops/q8ta_pixel_shuffle/q8ta_pixel_shuffle_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taPixelShuffleParams {
  uint32_t r;
  uint32_t out_c;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t in_c;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t numel;
  float input_scale;
  float inv_output_scale;
  int32_t input_zero_point;
  int32_t output_zero_point;
};
static_assert(
    sizeof(Q8taPixelShuffleParams) == 48,
    "Q8taPixelShuffleParams must match the WGSL Params struct (48 bytes)");

// int8 pixel_shuffle gather + requant; mirrors Vulkan q8ta_pixel_shuffle.glsl.
void q8ta_pixel_shuffle_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [x, in_scale, in_zp, out_inv_scale, out_zp, upscale_factor, out].
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_pixel_shuffle: in/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const size_t ndim = in_tensor.dims.size();
  if (ndim < 3 || out_tensor.dims.size() != ndim) {
    throw std::runtime_error("q8ta_pixel_shuffle: matching rank >= 3 expected");
  }
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_pixel_shuffle: null buffer binding");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));
  // 4th arg is already 1/output_scale (output_inv_scale); not re-inverted.
  const double output_inv_scale = graph.get_double(args.at(3));
  const int output_zero_point = graph.get_int(args.at(4));
  const int64_t r = graph.get_int(args.at(5));
  if (r < 1) {
    throw std::runtime_error("q8ta_pixel_shuffle: upscale_factor must be >= 1");
  }

  const uint32_t in_c = static_cast<uint32_t>(in_tensor.dims.at(ndim - 3));
  const uint32_t in_h = static_cast<uint32_t>(in_tensor.dims.at(ndim - 2));
  const uint32_t in_w = static_cast<uint32_t>(in_tensor.dims.at(ndim - 1));
  const uint32_t out_c = static_cast<uint32_t>(out_tensor.dims.at(ndim - 3));
  const uint32_t out_h = static_cast<uint32_t>(out_tensor.dims.at(ndim - 2));
  const uint32_t out_w = static_cast<uint32_t>(out_tensor.dims.at(ndim - 1));
  const uint32_t rr = static_cast<uint32_t>(r) * static_cast<uint32_t>(r);
  if (in_c % rr != 0) {
    throw std::runtime_error("q8ta_pixel_shuffle: in channels not div by r*r");
  }
  // The shader derives in c/h/w from the OUT dims, so an out shape inconsistent
  // with pixel_shuffle(in, r) (but same total numel) would read OOB. Mirrors
  // Vulkan Q8taPixelShuffle.cpp VK_CHECK_COND(in == out*r*r / out*r).
  if (out_c != in_c / rr || out_h != in_h * static_cast<uint32_t>(r) ||
      out_w != in_w * static_cast<uint32_t>(r)) {
    throw std::runtime_error(
        "q8ta_pixel_shuffle: out dims != pixel_shuffle(in, r)");
  }

  uint64_t numel = 1;
  for (int64_t d : out_tensor.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (numel == 0 || numel % 4 != 0) {
    throw std::runtime_error(
        "q8ta_pixel_shuffle: numel must be a nonzero multiple of 4");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("q8ta_pixel_shuffle: numel exceeds u32");
  }
  // in/out int8 (kernel clamps to [-128,127]) of equal numel.
  if (!in_tensor.is_int8 || !out_tensor.is_int8 || in_tensor.nbytes != numel ||
      out_tensor.nbytes != numel) {
    throw std::runtime_error(
        "q8ta_pixel_shuffle: in/out must be int8 of equal numel");
  }

  Q8taPixelShuffleParams params = {};
  params.r = static_cast<uint32_t>(r);
  params.out_c = out_c;
  params.out_h = out_h;
  params.out_w = out_w;
  params.in_c = in_c;
  params.in_h = in_h;
  params.in_w = in_w;
  params.numel = static_cast<uint32_t>(numel);
  params.input_scale = static_cast<float>(input_scale);
  params.inv_output_scale = static_cast<float>(output_inv_scale);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);

  const uint32_t num_words = static_cast<uint32_t>(numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taPixelShuffleWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_words, wg_size, "q8ta_pixel_shuffle");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taPixelShuffleParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taPixelShuffleParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {kQ8taPixelShuffleWGSL, WGPU_STRLEN};
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
  bg_entries[2].size = sizeof(Q8taPixelShuffleParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 3;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  const size_t dispatch_idx = graph.add_dispatch(
      {pipeline,
       bind_group,
       workgroup_count.x,
       "q8ta_pixel_shuffle",
       workgroup_count.y});

  // Dynamic shapes (supports_resize): recompute shape/numel/dispatch + UBO.
  Q8taPixelShuffleParams base = params;
  const uint32_t r_u = static_cast<uint32_t>(r);
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, base, r_u, wg_size, dispatch_idx, p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const size_t nd = d.size();
        if (nd < 3) {
          throw std::runtime_error("q8ta_pixel_shuffle(resize): rank < 3");
        }
        Q8taPixelShuffleParams p = base;
        p.in_c = static_cast<uint32_t>(d[nd - 3]);
        p.in_h = static_cast<uint32_t>(d[nd - 2]);
        p.in_w = static_cast<uint32_t>(d[nd - 1]);
        p.out_c = p.in_c / (r_u * r_u);
        p.out_h = p.in_h * r_u;
        p.out_w = p.in_w * r_u;
        std::vector<int64_t> out_d(d);
        out_d[nd - 3] = p.out_c;
        out_d[nd - 2] = p.out_h;
        out_d[nd - 1] = p.out_w;
        uint64_t n = 1;
        for (int64_t v : out_d) {
          n *= static_cast<uint64_t>(v);
        }
        if (n == 0 || n % 4 != 0 || n > UINT32_MAX) {
          throw std::runtime_error(
              "q8ta_pixel_shuffle(resize): numel must be a u32 multiple of 4");
        }
        p.numel = static_cast<uint32_t>(n);
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(),
            static_cast<uint32_t>(n / 4),
            wg_size,
            "q8ta_pixel_shuffle(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        g.set_cur_dims(out_id, out_d);
      });

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_pixel_shuffle.default, q8ta_pixel_shuffle_impl);
}

} // namespace executorch::backends::webgpu

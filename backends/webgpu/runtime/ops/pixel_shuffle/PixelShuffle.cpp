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
#include <executorch/backends/webgpu/runtime/ops/pixel_shuffle/pixel_shuffle_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct PixelShuffleParams {
  uint32_t r;
  uint32_t out_c;
  uint32_t out_h;
  uint32_t out_w;
  uint32_t in_c;
  uint32_t in_h;
  uint32_t in_w;
  uint32_t numel;
};
static_assert(
    sizeof(PixelShuffleParams) == 32,
    "PixelShuffleParams must match the WGSL Params struct (32 bytes)");

// pixel_shuffle: (N,C*r*r,H,W)->(N,C,H*r,W*r) rearrange (Vulkan glsl).
void pixel_shuffle_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [self, upscale_factor, out].
  const int in_id = args.at(0);
  const int r_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("pixel_shuffle: in/out arg is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const size_t ndim = in_tensor.dims.size();
  if (ndim < 3 || out_tensor.dims.size() != ndim) {
    throw std::runtime_error("pixel_shuffle: expected matching rank >= 3");
  }

  const int64_t r = graph.get_int(r_id);
  if (r < 1) {
    throw std::runtime_error("pixel_shuffle: upscale_factor must be >= 1");
  }

  // C/H/W are the last 3 dims; any leading dims collapse into the batch.
  const uint32_t in_c = static_cast<uint32_t>(in_tensor.dims.at(ndim - 3));
  const uint32_t in_h = static_cast<uint32_t>(in_tensor.dims.at(ndim - 2));
  const uint32_t in_w = static_cast<uint32_t>(in_tensor.dims.at(ndim - 1));
  const uint32_t out_c = static_cast<uint32_t>(out_tensor.dims.at(ndim - 3));
  const uint32_t out_h = static_cast<uint32_t>(out_tensor.dims.at(ndim - 2));
  const uint32_t out_w = static_cast<uint32_t>(out_tensor.dims.at(ndim - 1));
  // Mirror Vulkan VK_CHECK_COND(in_sizes[ndim-3] % (r*r) == 0).
  if (in_c % (static_cast<uint32_t>(r) * static_cast<uint32_t>(r)) != 0) {
    throw std::runtime_error("pixel_shuffle: in channels not divisible by r*r");
  }

  uint64_t out_numel = 1;
  for (int64_t d : out_tensor.dims) {
    out_numel *= static_cast<uint64_t>(d);
  }
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes != out_numel * sizeof(float)) {
    throw std::runtime_error("pixel_shuffle: non-4-byte operand (nbytes % 4)");
  }

  PixelShuffleParams params = {};
  params.r = static_cast<uint32_t>(r);
  params.out_c = out_c;
  params.out_h = out_h;
  params.out_w = out_w;
  params.in_c = in_c;
  params.in_h = in_h;
  params.in_w = in_w;
  params.numel = static_cast<uint32_t>(out_numel);

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kPixelShuffleWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(out_numel), wg_size, "pixel_shuffle");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(PixelShuffleParams));
  graph.add_uniform_buffer_bytes(sizeof(PixelShuffleParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kPixelShuffleWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           params_buf,
           sizeof(PixelShuffleParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "pixel_shuffle",
       workgroup_count.y});

  // Dynamic shapes: out = in last-3 rescaled by r; recompute params+dispatch.
  const uint32_t r_u = static_cast<uint32_t>(r);
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, r_u, wg_size, dispatch_idx, p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const size_t nd = d.size();
        if (nd < 3) {
          throw std::runtime_error("pixel_shuffle(resize): rank < 3");
        }
        PixelShuffleParams p = {};
        p.r = r_u;
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
        p.numel = static_cast<uint32_t>(n);
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), p.numel, wg_size, "pixel_shuffle(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        g.set_cur_dims(out_id, out_d);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.pixel_shuffle.default, pixel_shuffle_impl);
}

} // namespace executorch::backends::webgpu

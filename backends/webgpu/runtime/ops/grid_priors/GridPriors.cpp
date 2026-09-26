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
#include <executorch/backends/webgpu/runtime/ops/grid_priors/grid_priors_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct GridPriorsParams {
  uint32_t numel;
  uint32_t width;
  float stride;
  float offset;
};
static_assert(sizeof(GridPriorsParams) == 16, "GridPriorsParams must be 16 B");

// grid_priors: anchor grid-shifts [H*W,2] from H/W (Vulkan GridPriors.cpp).
void grid_priors_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [in, stride(int), offset(float), out]; only in's H/W are used.
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("grid_priors: in/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (out_tensor.buffer == nullptr) {
    throw std::runtime_error("grid_priors: null output buffer");
  }
  if (in_tensor.dims.size() < 2) {
    throw std::runtime_error("grid_priors: input must be at least 2D");
  }

  const int64_t stride = graph.get_int(args.at(1));
  const double offset = graph.get_double(args.at(2));
  const uint64_t height =
      static_cast<uint64_t>(in_tensor.dims.at(in_tensor.dims.size() - 2));
  const uint64_t width =
      static_cast<uint64_t>(in_tensor.dims.at(in_tensor.dims.size() - 1));
  const uint64_t numel = height * width * 2u;
  if (width == 0u || numel == 0u || numel > UINT32_MAX) {
    throw std::runtime_error("grid_priors: bad H/W (zero or numel > u32)");
  }
  // Output is fp32 [H*W, 2].
  if (out_tensor.is_int || out_tensor.nbytes != numel * sizeof(float)) {
    throw std::runtime_error("grid_priors: output must be fp32 [H*W, 2]");
  }

  GridPriorsParams params = {};
  params.numel = static_cast<uint32_t>(numel);
  params.width = static_cast<uint32_t>(width);
  params.stride = static_cast<float>(stride);
  params.offset = static_cast<float>(offset);

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kGridPriorsWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(numel), wg_size, "grid_priors");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(GridPriorsParams));
  graph.add_uniform_buffer_bytes(sizeof(GridPriorsParams));

  // Bind group: output (rw storage) + params (uniform); input data is unread.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kGridPriorsWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(GridPriorsParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "grid_priors",
       workgroup_count.y});

  // Dynamic shapes: recompute numel/width + dispatch + out dims [H*W, 2].
  WGPUBuffer params_buf = uniform_buffer;
  const float stride_f = static_cast<float>(stride);
  const float offset_f = static_cast<float>(offset);
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, stride_f, offset_f, wg_size, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.size() < 2) {
          throw std::runtime_error("grid_priors(resize): input < 2D");
        }
        const uint64_t h = static_cast<uint64_t>(d.at(d.size() - 2));
        const uint64_t w = static_cast<uint64_t>(d.at(d.size() - 1));
        const uint64_t n = h * w * 2u;
        if (w == 0u || n == 0u || n > UINT32_MAX) {
          throw std::runtime_error("grid_priors(resize): bad H/W");
        }
        GridPriorsParams p = {};
        p.numel = static_cast<uint32_t>(n);
        p.width = static_cast<uint32_t>(w);
        p.stride = stride_f;
        p.offset = offset_f;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), static_cast<uint32_t>(n), wg_size, "grid_priors");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        g.set_cur_dims(
            out_id, {static_cast<int64_t>(h * w), static_cast<int64_t>(2)});
      });

  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.grid_priors.default, grid_priors_impl);
}

} // namespace executorch::backends::webgpu

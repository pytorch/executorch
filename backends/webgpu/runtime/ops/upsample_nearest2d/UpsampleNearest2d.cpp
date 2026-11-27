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
#include <executorch/backends/webgpu/runtime/ops/upsample_nearest2d/upsample_nearest2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct UpsampleParams {
  uint32_t N;
  uint32_t C;
  uint32_t IH;
  uint32_t IW;
  uint32_t OH;
  uint32_t OW;
  uint32_t _p0;
  uint32_t _p1;
};
static_assert(sizeof(UpsampleParams) == 32, "UpsampleParams must be 32 bytes");

// OH/OW come from the output tensor's own dims, not the size/scale args.
void upsample_nearest2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 2) {
    throw std::runtime_error("WebGPU upsample_nearest2d: expected >=2 args");
  }
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error("WebGPU upsample_nearest2d: expected 4D in/out");
  }

  const uint32_t N = static_cast<uint32_t>(in.dims[0]);
  const uint32_t C = static_cast<uint32_t>(in.dims[1]);
  const uint32_t IH = static_cast<uint32_t>(in.dims[2]);
  const uint32_t IW = static_cast<uint32_t>(in.dims[3]);
  const uint32_t OH = static_cast<uint32_t>(out.dims[2]);
  const uint32_t OW = static_cast<uint32_t>(out.dims[3]);

  if (static_cast<uint32_t>(out.dims[0]) != N ||
      static_cast<uint32_t>(out.dims[1]) != C) {
    throw std::runtime_error("WebGPU upsample_nearest2d: N/C mismatch");
  }
  if (IH == 0 || IW == 0 || OH == 0 || OW == 0) {
    throw std::runtime_error("WebGPU upsample_nearest2d: zero spatial dim");
  }

  uint64_t out_numel = utils::check_fp32(out, "upsample_nearest2d", "output");

  // Up-front (throw before any buffer alloc -> no leak-on-throw).
  // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
  // ceiling. stride_x lets the shader decode idx = gid.y*stride_x + gid.x.
  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      utils::checked_u32(out_numel, "upsample_nearest2d"),
      kUpsampleNearest2dWorkgroupSizeX,
      "upsample_nearest2d");

  UpsampleParams params = {};
  params.N = N;
  params.C = C;
  params.IH = IH;
  params.IW = IW;
  params.OH = OH;
  params.OW = OW;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(UpsampleParams));
  graph.add_uniform_buffer_bytes(sizeof(UpsampleParams));

  auto constants = utils::make_grid_constants(grid);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kUpsampleNearest2dWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(UpsampleParams)},
      },
      constants.data(),
      constants.size());

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, grid.count_x, grid.count_y);

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.upsample_nearest2d.vec, upsample_nearest2d_impl);
}

} // namespace executorch::backends::webgpu

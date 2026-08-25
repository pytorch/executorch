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
#include <executorch/backends/webgpu/runtime/ops/upsample_bilinear2d/upsample_bilinear2d_wgsl.h>

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
  uint32_t align_corners;
  uint32_t _p0;
};
static_assert(sizeof(UpsampleParams) == 32, "UpsampleParams must be 32 bytes");

// aten.upsample_bilinear2d.vec: bilinear NCHW resize; align_corners src-index.
void upsample_bilinear2d_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args) {
  if (args.size() < 3) {
    throw std::runtime_error("WebGPU upsample_bilinear2d: expected >=3 args");
  }
  const int in_id = args.at(0);
  const int align_corners_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error("WebGPU upsample_bilinear2d: expected 4D in/out");
  }

  const uint32_t N = static_cast<uint32_t>(in.dims[0]);
  const uint32_t C = static_cast<uint32_t>(in.dims[1]);
  const uint32_t IH = static_cast<uint32_t>(in.dims[2]);
  const uint32_t IW = static_cast<uint32_t>(in.dims[3]);
  const uint32_t OH = static_cast<uint32_t>(out.dims[2]);
  const uint32_t OW = static_cast<uint32_t>(out.dims[3]);

  if (static_cast<uint32_t>(out.dims[0]) != N ||
      static_cast<uint32_t>(out.dims[1]) != C) {
    throw std::runtime_error("WebGPU upsample_bilinear2d: N/C mismatch");
  }
  if (IH == 0 || IW == 0 || OH == 0 || OW == 0) {
    throw std::runtime_error("WebGPU upsample_bilinear2d: zero spatial dim");
  }

  uint64_t out_numel = uint64_t(N) * C * OH * OW;
  if (out.nbytes != out_numel * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU upsample_bilinear2d: fp32-only (byte mismatch)");
  }
  if (out_numel > 0xFFFFFFFFull) {
    throw std::runtime_error("WebGPU upsample_bilinear2d: output too large");
  }

  const uint32_t align_corners = graph.get_bool(align_corners_id) ? 1u : 0u;

  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      static_cast<uint32_t>(out_numel),
      kUpsampleBilinear2dWorkgroupSizeX,
      "upsample_bilinear2d");

  UpsampleParams params = {};
  params.N = N;
  params.C = C;
  params.IH = IH;
  params.IW = IW;
  params.OH = OH;
  params.OW = OW;
  params.align_corners = align_corners;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(UpsampleParams));
  graph.add_uniform_buffer_bytes(sizeof(UpsampleParams));

  auto constants = utils::make_grid_constants(grid);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kUpsampleBilinear2dWGSL,
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

  WebGPUDispatch dispatch{};
  dispatch.pipeline = bundle.pipeline;
  dispatch.bind_group = bundle.bind_group;
  dispatch.workgroup_count_x = grid.count_x;
  dispatch.workgroup_count_y = grid.count_y;
  graph.add_dispatch(dispatch);

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.upsample_bilinear2d.vec, upsample_bilinear2d_impl);
}

} // namespace executorch::backends::webgpu

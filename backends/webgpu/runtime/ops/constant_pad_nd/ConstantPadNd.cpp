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
#include <executorch/backends/webgpu/runtime/ops/constant_pad_nd/constant_pad_nd_wgsl.h>

#include <webgpu/webgpu.h>

#include <array>
#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct PadParams {
  uint32_t out_dims[4];
  uint32_t in_dims[4];
  uint32_t left[4];
  uint32_t out_numel;
  float value;
  uint32_t _p0;
  uint32_t _p1;
};
static_assert(sizeof(PadParams) == 64, "PadParams must be 64 bytes");

// `pad` is reversed-dim (last-dim-first pairs); output right-aligned into 4D.
void constant_pad_nd_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 3) {
    throw std::runtime_error("WebGPU constant_pad_nd: expected >=3 args");
  }
  const int in_id = args.at(0);
  const int pad_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(out_id);

  const size_t nd = in.dims.size();
  if (nd == 0 || nd > 4) {
    throw std::runtime_error("WebGPU constant_pad_nd: rank must be 1..4");
  }
  if (out.dims.size() != nd) {
    throw std::runtime_error("WebGPU constant_pad_nd: in/out rank mismatch");
  }

  if (graph.get_value_type(pad_id) != WebGPUGraph::ValueType::IntList) {
    throw std::runtime_error("WebGPU constant_pad_nd: pad is not an IntList");
  }
  const std::vector<int64_t>& pad = graph.get_int_list(pad_id);
  if (pad.size() % 2 != 0) {
    throw std::runtime_error("WebGPU constant_pad_nd: pad must be even-length");
  }
  const size_t npad = pad.size() / 2;
  if (npad > nd) {
    throw std::runtime_error("WebGPU constant_pad_nd: pad longer than rank");
  }

  // value scalar (default 0). Vulkan serializes a Scalar as Int or Double.
  float value = 0.0f;
  if (args.size() >= 4) {
    value = utils::scalar_or(graph, args.at(2), 0.0f);
  }

  // Per-dim left/right pad (pad list is reversed-dim, from the LAST dim).
  std::array<int64_t, 4> left = {0, 0, 0, 0};
  std::array<int64_t, 4> right = {0, 0, 0, 0};
  for (size_t k = 0; k < npad; k++) {
    const size_t d = nd - 1 - k; // k-th pad entry -> dim (nd-1-k)
    left[d] = pad[2 * k];
    right[d] = pad[2 * k + 1];
  }

  // Validate output dims == in + left + right per dim (loud-fail on a wrong
  // pad-list interpretation), before any buffer alloc -> no leak-on-throw.
  for (size_t d = 0; d < nd; d++) {
    const int64_t expect = in.dims[d] + left[d] + right[d];
    if (expect < 0 || static_cast<int64_t>(out.dims[d]) != expect) {
      throw std::runtime_error("WebGPU constant_pad_nd: output shape mismatch");
    }
  }

  const uint64_t out_numel =
      utils::check_fp32(out, "constant_pad_nd", "output");

  // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
  // ceiling. stride_x lets the shader decode idx = gid.y*stride_x + gid.x.
  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      utils::checked_u32(out_numel, "constant_pad_nd"),
      kConstantPadNdWorkgroupSizeX,
      "constant_pad_nd");

  // Right-align dims into [4]: leading (4-nd) entries get extent 1, pad 0.
  PadParams params = {};
  for (int s = 0; s < 4; s++) {
    params.out_dims[s] = 1;
    params.in_dims[s] = 1;
    params.left[s] = 0;
  }
  const size_t off = 4 - nd;
  for (size_t d = 0; d < nd; d++) {
    params.out_dims[off + d] = static_cast<uint32_t>(out.dims[d]);
    params.in_dims[off + d] = static_cast<uint32_t>(in.dims[d]);
    params.left[off + d] = static_cast<uint32_t>(left[d]);
  }
  params.out_numel = static_cast<uint32_t>(out_numel);
  params.value = value;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(PadParams));
  graph.add_uniform_buffer_bytes(sizeof(PadParams));

  auto constants = utils::make_grid_constants(grid);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kConstantPadNdWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2, WGPUBufferBindingType_Uniform, uniform_buffer, sizeof(PadParams)},
      },
      constants.data(),
      constants.size());

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, grid.count_x, grid.count_y);

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.constant_pad_nd.default, constant_pad_nd_impl);
}

} // namespace executorch::backends::webgpu

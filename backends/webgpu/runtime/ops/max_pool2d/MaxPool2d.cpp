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
#include <executorch/backends/webgpu/runtime/ops/max_pool2d/max_pool2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct PoolParams {
  uint32_t N, C, IH, IW, OH, OW;
  uint32_t kH, kW, sH, sW, pH, pW, dH, dW;
  uint32_t write_indices, _p1;
};
static_assert(sizeof(PoolParams) == 64, "PoolParams must be 64 bytes");

// out_list = ValueList[values, indices] (mirrors Vulkan Pool.cpp).
void max_pool2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 6) {
    throw std::runtime_error("WebGPU max_pool2d: expected >=6 args");
  }
  const int in_id = args.at(0);
  const int kernel_id = args.at(1);
  const int stride_id = args.at(2);
  const int padding_id = args.at(3);
  const int dilation_id = args.at(4);
  const int out_list_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  if (graph.get_value_type(out_list_id) != WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error("WebGPU max_pool2d: out is not a ValueList");
  }
  const std::vector<int>& outs = graph.get_value_list(out_list_id);
  if (outs.empty()) {
    throw std::runtime_error("WebGPU max_pool2d: empty out ValueList");
  }
  const int values_id = outs.at(0); // [0]=values, [1]=indices
  const bool has_indices = outs.size() > 1 &&
      graph.get_value_type(outs.at(1)) == WebGPUGraph::ValueType::Tensor;
  if (has_indices) {
    const auto& idx_t = graph.get_tensor(outs.at(1));
    if (idx_t.dims != graph.get_tensor(values_id).dims) {
      throw std::runtime_error(
          "WebGPU max_pool2d: indices output shape must match values output");
    }
    // The WGSL kernel writes int32 indices (mirrors Vulkan's Pool.cpp, which
    // also computes/stores 32-bit indices despite ATen's int64 schema) — throw
    // rather than silently mis-stride the buffer if this graph's indices
    // tensor ever turns out to be int64 (8 bytes/elem) instead.
    const uint64_t idx_numel = utils::numel_of(idx_t.dims);
    if (idx_t.nbytes != idx_numel * sizeof(int32_t)) {
      throw std::runtime_error(
          "WebGPU max_pool2d: indices output must be int32 (4 bytes/elem); "
          "got a different byte width, would mis-stride the i32 kernel write");
    }
  }

  const auto& in = graph.get_tensor(in_id);
  const auto& out = graph.get_tensor(values_id);
  if (in.dims.size() != 4 || out.dims.size() != 4) {
    throw std::runtime_error("WebGPU max_pool2d: expected 4D in/out");
  }

  // kernel/stride/padding/dilation are int lists; PyTorch broadcasts a single
  // value to both spatial dims. stride defaults to kernel_size when empty.
  uint32_t kH, kW, sH, sW, pH, pW, dH, dW;
  utils::parse_hw(
      graph.get_int_list(kernel_id), kH, kW, "max_pool2d", "kernel_size");
  const std::vector<int64_t> stride_v = graph.get_int_list(stride_id);
  if (stride_v.empty()) {
    sH = kH;
    sW = kW;
  } else {
    utils::parse_hw(stride_v, sH, sW, "max_pool2d", "stride");
  }
  utils::parse_hw(
      graph.get_int_list(padding_id), pH, pW, "max_pool2d", "padding");
  utils::parse_hw(
      graph.get_int_list(dilation_id), dH, dW, "max_pool2d", "dilation");

  const uint32_t N = static_cast<uint32_t>(in.dims[0]);
  const uint32_t C = static_cast<uint32_t>(in.dims[1]);
  const uint32_t IH = static_cast<uint32_t>(in.dims[2]);
  const uint32_t IW = static_cast<uint32_t>(in.dims[3]);
  if (sH == 0 || sW == 0) {
    throw std::runtime_error("WebGPU max_pool2d: zero stride");
  }

  const int64_t oh_num =
      int64_t(IH) + 2 * int64_t(pH) - int64_t(dH) * (int64_t(kH) - 1) - 1;
  const int64_t ow_num =
      int64_t(IW) + 2 * int64_t(pW) - int64_t(dW) * (int64_t(kW) - 1) - 1;
  if (oh_num < 0 || ow_num < 0) {
    throw std::runtime_error("WebGPU max_pool2d: kernel larger than input");
  }
  const uint32_t OH = static_cast<uint32_t>(oh_num / int64_t(sH)) + 1;
  const uint32_t OW = static_cast<uint32_t>(ow_num / int64_t(sW)) + 1;

  // Validate against the serialized values output [N, C, OH, OW] (loud-fail if
  // the arg interpretation is wrong, e.g. ceil_mode or a different layout).
  if (static_cast<uint32_t>(out.dims[0]) != N ||
      static_cast<uint32_t>(out.dims[1]) != C ||
      static_cast<uint32_t>(out.dims[2]) != OH ||
      static_cast<uint32_t>(out.dims[3]) != OW) {
    throw std::runtime_error("WebGPU max_pool2d: output shape mismatch");
  }

  uint64_t out_numel = utils::check_fp32(out, "max_pool2d", "output");

  // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
  // ceiling. stride_x lets the shader decode idx = gid.y*stride_x + gid.x.
  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device,
      utils::checked_u32(out_numel, "max_pool2d"),
      kMaxPool2dWorkgroupSizeX,
      "max_pool2d");

  PoolParams params = {};
  params.N = N;
  params.C = C;
  params.IH = IH;
  params.IW = IW;
  params.OH = OH;
  params.OW = OW;
  params.kH = kH;
  params.kW = kW;
  params.sH = sH;
  params.sW = sW;
  params.pH = pH;
  params.pW = pW;
  params.dH = dH;
  params.dW = dW;
  params.write_indices = has_indices ? 1u : 0u;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(PoolParams));
  graph.add_uniform_buffer_bytes(sizeof(PoolParams));

  auto grid_constants = utils::make_grid_constants(grid);

  // write_indices==0 -> the shader never stores to out_idx, so a tiny dummy
  // buffer is safe (mirrors NativeLayerNorm.cpp's dummy_affine pattern).
  utils::OptionalBinding idx = utils::make_optional_binding(
      device,
      has_indices,
      has_indices ? graph.get_tensor(outs.at(1)).buffer : nullptr,
      has_indices ? graph.get_tensor(outs.at(1)).nbytes : 0);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kMaxPool2dWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(PoolParams)},
          {3, WGPUBufferBindingType_Storage, idx.buffer, idx.nbytes},
      },
      grid_constants.data(),
      grid_constants.size());

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, grid.count_x, grid.count_y);

  wgpuBufferRelease(uniform_buffer);
  if (idx.owned_dummy != nullptr) {
    wgpuBufferRelease(idx.owned_dummy);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.max_pool2d_with_indices.default, max_pool2d_impl);
}

} // namespace executorch::backends::webgpu

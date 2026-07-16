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
#include <executorch/backends/webgpu/runtime/ops/linear_fp32/linear_fp32_tiled_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/linear_fp32/linear_fp32_vec4_wgsl.h>

#include <webgpu/webgpu.h>

#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct LinearParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t has_bias;
};
static_assert(sizeof(LinearParams) == 16, "LinearParams must be 16 bytes");

constexpr uint32_t kTile = 32u;

// aten.linear.default args: [in, weight, bias, out] (mirrors Vulkan Linear.cpp
// linear_packed_weight). Shared-memory-tiled GEMM (+bias), vec4-over-K when
// K%4==0 — mirrors the sibling `linear` op's linear_tiled.wgsl/linear_vec4.wgsl
// (Linear.cpp:95,98), with a bias epilogue added. weight is the prepacked
// [N, K] constant (et_vk.prepack copies it). bias may be None.
void linear_fp32_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int bias_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& out_tensor = graph.get_tensor(out_id);

  if (in_tensor.dims.empty() || out_tensor.dims.empty()) {
    throw std::runtime_error("WebGPU linear: empty input/output dims");
  }

  const uint64_t out_numel = utils::check_fp32(out_tensor, "linear", "output");

  const uint32_t N = static_cast<uint32_t>(out_tensor.dims.back());
  const uint32_t K = static_cast<uint32_t>(in_tensor.dims.back());
  if (N == 0 || K == 0) {
    throw std::runtime_error("WebGPU linear: zero N or K");
  }
  const uint32_t M = static_cast<uint32_t>(out_numel / N);

  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  // A genuinely-2D tile dispatch (mirrors Linear.cpp) doesn't need the 1D-flat
  // stride_x recovery trick; throw before any allocation if it can't fit.
  utils::WgCount tile_grid =
      utils::compute_tile_grid_2d(device, N, M, kTile, "linear_fp32");
  const bool use_vec4 = (K % 4u == 0u);

  LinearParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.has_bias = has_bias ? 1u : 0u;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(LinearParams));
  graph.add_uniform_buffer_bytes(sizeof(LinearParams));

  // A 4-byte dummy storage buffer to satisfy the t_bias binding when bias is
  // None; the shader never reads it (has_bias gate).
  utils::OptionalBinding bias = utils::make_optional_binding(
      device,
      has_bias,
      has_bias ? graph.get_tensor(bias_id).buffer : nullptr,
      has_bias ? graph.get_tensor(bias_id).nbytes : 0);

  // Tiled kernels have a fixed @workgroup_size(8, 8, 1) — no override constant.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      use_vec4 ? kLinearFp32Vec4WGSL : kLinearFp32TiledWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight_tensor.buffer,
           weight_tensor.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, bias.buffer, bias.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(LinearParams)},
      });

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, tile_grid.x, tile_grid.y);

  wgpuBufferRelease(uniform_buffer);
  if (bias.owned_dummy != nullptr) {
    wgpuBufferRelease(bias.owned_dummy);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.linear.default, linear_fp32_impl);
}

} // namespace executorch::backends::webgpu

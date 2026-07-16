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
#include <executorch/backends/webgpu/runtime/ops/addmm/addmm_tiled_wgsl.h>

#include <webgpu/webgpu.h>

#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct AddmmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t self_2d;
  float beta;
  float alpha;
  uint32_t _pad[2];
};
static_assert(sizeof(AddmmParams) == 32, "AddmmParams must be 32 bytes");

constexpr uint32_t kTile = 32u;

// aten.addmm.default args: [self, mat1, mat2, beta, alpha, out].
// out = beta*self + alpha*(mat1 @ mat2); mat1 [M,K], mat2 [K,N], self [N] or
// [M,N] (HF Linear lowers to addmm with a [N] bias). Shared-memory-tiled GEMM,
// re-derived for mat2's [K,N] layout (mirrors the sibling `linear` op's
// linear_tiled.wgsl skeleton, NOT its [N,K]-shaped read_b). No vec4 variant:
// unlike linear's weight [N,K] (K contiguous, vec4-over-K natural on both
// sides), mat2 [K,N] has N contiguous — vec4-over-K would need a strided
// gather on the mat2 side, eroding the benefit; the tiled kernel alone already
// solves the coalescing gap via the cooperative shared-memory tile load.
void addmm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int self_id = args.at(0);
  const int mat1_id = args.at(1);
  const int mat2_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& self_t = graph.get_tensor(self_id);
  const auto& mat1 = graph.get_tensor(mat1_id);
  const auto& mat2 = graph.get_tensor(mat2_id);
  const auto& out = graph.get_tensor(out_id);

  if (self_t.buffer == nullptr || mat1.buffer == nullptr ||
      mat2.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("WebGPU addmm: null buffer binding");
  }
  if (mat1.dims.size() != 2 || mat2.dims.size() != 2 || out.dims.size() != 2) {
    throw std::runtime_error("WebGPU addmm: expected 2D self-mm/mat1/mat2/out");
  }

  const uint32_t M = static_cast<uint32_t>(out.dims[0]);
  const uint32_t N = static_cast<uint32_t>(out.dims[1]);
  const uint32_t K = static_cast<uint32_t>(mat1.dims[1]);
  if (M == 0 || N == 0 || K == 0) {
    throw std::runtime_error("WebGPU addmm: zero M/N/K");
  }
  if (static_cast<uint32_t>(mat2.dims[0]) != K ||
      static_cast<uint32_t>(mat2.dims[1]) != N) {
    throw std::runtime_error("WebGPU addmm: mat2 shape != [K, N]");
  }
  const uint64_t out_numel = utils::check_fp32(out, "addmm", "output");

  const uint64_t self_numel = utils::numel(self_t.dims);
  const bool self_2d = self_numel == out_numel;
  if (!self_2d && self_numel != N) {
    throw std::runtime_error(
        "WebGPU addmm: self must broadcast from [N] or [M,N]");
  }

  const float beta = utils::scalar_or(graph, args.at(3), 1.0f);
  const float alpha = utils::scalar_or(graph, args.at(4), 1.0f);

  // A genuinely-2D tile dispatch doesn't need the 1D-flat stride_x recovery
  // trick; throw before any allocation if it can't fit.
  utils::WgCount tile_grid =
      utils::compute_tile_grid_2d(device, N, M, kTile, "addmm");

  AddmmParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.self_2d = self_2d ? 1u : 0u;
  params.beta = beta;
  params.alpha = alpha;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(AddmmParams));
  graph.add_uniform_buffer_bytes(sizeof(AddmmParams));

  // Tiled kernel has a fixed @workgroup_size(8, 8, 1) — no override constant.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kAddmmTiledWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           self_t.buffer,
           self_t.nbytes},
          {2, WGPUBufferBindingType_ReadOnlyStorage, mat1.buffer, mat1.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, mat2.buffer, mat2.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(AddmmParams)},
      });

  graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, tile_grid.x, tile_grid.y);

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.addmm.default, addmm_impl);
}

} // namespace executorch::backends::webgpu

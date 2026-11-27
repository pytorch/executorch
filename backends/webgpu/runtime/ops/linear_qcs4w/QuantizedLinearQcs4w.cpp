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
#include <executorch/backends/webgpu/runtime/ops/linear_qcs4w/qcs4w_linear_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct; 16-byte aligned.
struct Qcs4wParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
};
static_assert(sizeof(Qcs4wParams) == 16, "Qcs4wParams must be 16 bytes");

// Register-tile dims; MUST match TM/TN in qcs4w_linear.wgsl.
constexpr int64_t kQcs4wTileM = 4;
constexpr int64_t kQcs4wTileN = 4;

utils::WgCount compute_qcs4w_workgroup_count(
    WGPUDevice device,
    uint32_t m,
    uint32_t n,
    uint32_t wg_size) {
  const int64_t total_tiles = utils::div_up<int64_t>(m, kQcs4wTileM) *
      utils::div_up<int64_t>(n, kQcs4wTileN);
  if (total_tiles > static_cast<int64_t>(UINT32_MAX)) {
    throw std::runtime_error("WebGPU linear_qcs4w: tile count exceeds u32");
  }
  return utils::compute_2d_workgroup_count(
      device, static_cast<uint32_t>(total_tiles), wg_size, "linear_qcs4w");
}

// linear_qcs4w: 4-bit channels-symmetric weight, per-channel scale (no bias).
void qcs4w_linear_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int scales_id = args.at(2);
  const int out_id = args.at(args.size() - 1);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& out = graph.get_tensor(out_id);
  if (in.buffer == nullptr || weight.buffer == nullptr ||
      scales.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("WebGPU linear_qcs4w: null buffer binding");
  }
  if (in.dims.empty() || weight.dims.size() < 2 || scales.dims.empty()) {
    throw std::runtime_error("WebGPU linear_qcs4w: malformed input dims");
  }

  const uint32_t K = static_cast<uint32_t>(in.dims.back());
  if (K == 0) {
    throw std::runtime_error("WebGPU linear_qcs4w: K == 0");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  if (in_numel % K != 0) {
    throw std::runtime_error(
        "WebGPU linear_qcs4w: input numel not a multiple of K");
  }
  const uint32_t M = static_cast<uint32_t>(in_numel / K);
  const uint32_t N = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t K_packed = static_cast<uint32_t>(weight.dims[1]);
  if (M == 0 || N == 0) {
    throw std::runtime_error("WebGPU linear_qcs4w: M or N == 0");
  }
  // int4 packing is 2 nibbles/byte, so K_packed must be ceil(K/2) (guards OOB).
  if (K_packed != (K + 1) / 2) {
    throw std::runtime_error("WebGPU linear_qcs4w: K_packed must be ceil(K/2)");
  }
  // Weight is read as array<u32>; a non-multiple-of-4 byte count over-reads.
  if ((static_cast<uint64_t>(N) * K_packed) % 4u != 0u) {
    throw std::runtime_error(
        "WebGPU linear_qcs4w: N*K_packed must be a multiple of 4 (u32-packed)");
  }

  // fp32-only byte-size guards; scales is per-output-channel (1D, N entries).
  const uint64_t weight_numel =
      static_cast<uint64_t>(N) * static_cast<uint64_t>(K_packed);
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != static_cast<uint64_t>(M) * N * sizeof(float) ||
      scales.nbytes < static_cast<uint64_t>(N) * sizeof(float) ||
      weight.nbytes != weight_numel) {
    throw std::runtime_error(
        "WebGPU linear_qcs4w: fp32-only (byte-size mismatch)");
  }

  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQcs4wLinearWorkgroupSizeX);
  const utils::WgCount workgroup_count =
      compute_qcs4w_workgroup_count(device, M, N, wg_size);

  Qcs4wParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(Qcs4wParams));
  graph.add_uniform_buffer_bytes(sizeof(Qcs4wParams));

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kQcs4wLinearWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight.buffer,
           weight.nbytes},
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           scales.buffer,
           scales.nbytes},
          {4,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(Qcs4wParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "linear_qcs4w",
       workgroup_count.y});

  // Dynamic shapes: recompute dispatch + params.M for the live M.
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, M, K, N, K_packed, wg_size, dispatch_idx, uniform_buffer](
          WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        if (d.empty()) {
          throw std::runtime_error(
              "WebGPU linear_qcs4w(resize): empty input dims");
        }
        if (static_cast<uint32_t>(d.back()) != K) {
          throw std::runtime_error(
              "WebGPU linear_qcs4w(resize): last dim must equal K");
        }
        const uint64_t numel = utils::numel_of(d);
        if (numel % static_cast<uint64_t>(K) != 0u) {
          throw std::runtime_error(
              "WebGPU linear_qcs4w(resize): live numel not a multiple of K");
        }
        const uint32_t m =
            static_cast<uint32_t>(numel / static_cast<uint64_t>(K));
        if (m == 0u || m > M) {
          throw std::runtime_error(
              "WebGPU linear_qcs4w(resize): live M is 0 or exceeds build max");
        }
        const utils::WgCount wgc =
            compute_qcs4w_workgroup_count(g.device(), m, N, wg_size);
        Qcs4wParams p = {};
        p.M = m;
        p.N = N;
        p.K = K;
        p.K_packed = K_packed;
        wgpuQueueWriteBuffer(g.queue(), uniform_buffer, 0, &p, sizeof(p));
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        std::vector<int64_t> od(d.begin(), d.end());
        od.back() = static_cast<int64_t>(N);
        g.set_cur_dims(out_id, od);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_qcs4w.default, qcs4w_linear_impl);
}

} // namespace executorch::backends::webgpu

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
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_av_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_qk_entry_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_qk_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa/sdpa_softmax_wgsl.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct QkParams {
  uint32_t B;
  uint32_t H;
  uint32_t S_q;
  uint32_t S_kv;
  uint32_t D;
  uint32_t has_mask;
  uint32_t _pad0;
  float scale;
};
static_assert(sizeof(QkParams) == 32, "QkParams must be 32 bytes");

struct AvParams {
  uint32_t B;
  uint32_t H;
  uint32_t S_q;
  uint32_t S_kv;
  uint32_t D;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(sizeof(AvParams) == 32, "AvParams must be 32 bytes");

// Mirrors the Params struct in sdpa_softmax.wgsl (file-local in Sdpa.cpp, so
// re-declared here for the reuse).
struct SoftmaxParams {
  uint32_t num_rows;
  uint32_t row_width;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(SoftmaxParams) == 16, "SoftmaxParams must be 16 bytes");

// aten op: et_vk.sdpa.default. Args: [q, k, v, attn_mask, scale, out] (mirrors
// Vulkan fused_sdpa_impl, SDPA.cpp). Non-causal, no KV-cache; all tensors are
// DSHB [B, H, S, D], row-major. Three dispatches: QK (scaled, optional additive
// mask) -> softmax (reused sdpa_softmax.wgsl) -> AV.
void et_vk_sdpa_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() != 6) {
    throw std::runtime_error("WebGPU et_vk.sdpa: expected 6 args");
  }
  const int q_id = args.at(0);
  const int k_id = args.at(1);
  const int v_id = args.at(2);
  const int mask_id = args.at(3);
  const int scale_id = args.at(4);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();

  const auto& q = graph.get_tensor(q_id);
  const auto& k = graph.get_tensor(k_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& out = graph.get_tensor(out_id);

  if (q.dims.size() < 3) {
    throw std::runtime_error("WebGPU et_vk.sdpa: q rank < 3");
  }
  const int rank = static_cast<int>(q.dims.size());
  const uint32_t D = static_cast<uint32_t>(q.dims[rank - 1]);
  const uint32_t S_q = static_cast<uint32_t>(q.dims[rank - 2]);
  const uint32_t H = static_cast<uint32_t>(q.dims[rank - 3]);
  if (D == 0 || S_q == 0 || H == 0) {
    throw std::runtime_error("WebGPU et_vk.sdpa: zero D/S_q/H");
  }
  // QK/AV kernels view q/k/v/out as vec4<f32> over D; every model in scope
  // (Whisper/Voxtral/DaViT/BART/Hiera) uses D=64 or 128, always %4==0.
  utils::check_vec4_aligned(D, "et_vk.sdpa", "D");
  const uint64_t q_numel = utils::check_fp32(q, "et_vk.sdpa", "q");
  const uint32_t B = static_cast<uint32_t>(q_numel / (uint64_t(H) * S_q * D));

  // Asymmetric seq supported (S_q != S_kv, e.g. Hiera pooled query): k/v carry
  // S_kv, q carries S_q; B/H/D must match across q/k/v. out is [B, H, S_q, D].
  // When S_q == S_kv this is plain self-attention (bit-identical to before).
  if (k.dims != v.dims) {
    throw std::runtime_error("WebGPU et_vk.sdpa: k/v shape mismatch");
  }
  if (k.dims.size() != q.dims.size()) {
    throw std::runtime_error("WebGPU et_vk.sdpa: q/k rank mismatch");
  }
  if (static_cast<uint32_t>(k.dims[rank - 1]) != D ||
      static_cast<uint32_t>(k.dims[rank - 3]) != H) {
    throw std::runtime_error("WebGPU et_vk.sdpa: q/k/v must share H and D");
  }
  const uint32_t S_kv = static_cast<uint32_t>(k.dims[rank - 2]);
  if (S_kv == 0) {
    throw std::runtime_error("WebGPU et_vk.sdpa: zero S_kv");
  }
  // Leading (batch) dims must agree across q/k/v.
  for (int d = 0; d < rank - 3; ++d) {
    if (k.dims[d] != q.dims[d]) {
      throw std::runtime_error("WebGPU et_vk.sdpa: q/k batch dims mismatch");
    }
  }
  // out must be [B, H, S_q, D] (same as q's shape).
  if (out.dims != q.dims) {
    throw std::runtime_error(
        "WebGPU et_vk.sdpa: out shape must match q [B, H, S_q, D]");
  }

  const bool has_mask =
      graph.get_value_type(mask_id) == WebGPUGraph::ValueType::Tensor;
  if (has_mask) {
    // The QK shader indexes mask as [B, H, S_q, S_kv] row-major; require it.
    const auto& mask = graph.get_tensor(mask_id);
    if (mask.nbytes != uint64_t(B) * H * S_q * S_kv * sizeof(float)) {
      throw std::runtime_error(
          "WebGPU et_vk.sdpa: attn_mask must be [B, H, S_q, S_kv] fp32");
    }
  }

  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto scale_type = graph.get_value_type(scale_id);
  if (scale_type == WebGPUGraph::ValueType::Double) {
    scale = static_cast<float>(graph.get_double(scale_id));
  } else if (scale_type != WebGPUGraph::ValueType::Null) {
    throw std::runtime_error("WebGPU et_vk.sdpa: scale must be Double or None");
  }

  const uint64_t num_rows = uint64_t(B) * H * S_q; // attn_weights rows
  const uint64_t aw_numel = num_rows * S_kv; // [B, H, S_q, S_kv]
  const uint64_t out_numel = uint64_t(B) * H * S_q * D;
  const size_t aw_bytes = static_cast<size_t>(aw_numel) * sizeof(float);

  // Up-front dispatch-limit checks (throw BEFORE any buffer alloc → no leak).
  // QK: per-row (one thread per (b,h,s) row, vec4 loads) is fastest for
  // standard attention, but starves the GPU on channel attention (S_q =
  // head_dim, so num_rows is tiny → few workgroups serial over a huge S_kv*D).
  // Route to the per-entry kernel (one thread per (b,h,s,c) attn entry,
  // 2D-folded) below an occupancy floor; both write a layout-identical
  // attn[B,H,S_q,S_kv] so softmax/AV are unchanged, and either branch is
  // numerically correct, so the floor is a perf knob only (Canary M4 Pro:
  // per-entry ~15-30x faster at num_rows <= 256, per-row wins at num_rows >=
  // 8192). AV = one per (b,h,s,d4) vec4; softmax = one workgroup per row.
  constexpr uint32_t kQkEntryOccupancyFloor = 4096u;
  const bool qk_per_entry = num_rows < kQkEntryOccupancyFloor;
  const uint32_t qk_wg_size = utils::clamp_workgroup_size(
      device,
      qk_per_entry ? kEtVkSdpaQkEntryWorkgroupSizeX
                   : kEtVkSdpaQkWorkgroupSizeX);
  uint32_t qk_wg_count = 0;
  utils::WgCount qk_entry_grid = {};
  if (qk_per_entry) {
    qk_entry_grid = utils::compute_2d_workgroup_count(
        device,
        static_cast<uint32_t>(aw_numel),
        qk_wg_size,
        "et_vk_sdpa_qk_entry");
  } else {
    qk_wg_count = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(num_rows), qk_wg_size, "et_vk_sdpa_qk");
  }
  const uint32_t av_wg_size =
      utils::clamp_workgroup_size(device, kEtVkSdpaAvWorkgroupSizeX);
  const uint64_t out_numel4 =
      out_numel / 4; // exact: D % 4 == 0 (checked above)
  const uint32_t av_wg_count = utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(out_numel4), av_wg_size, "et_vk_sdpa_av");
  // Near-square 2D grid of workgroups (1 workgroup = 1 row) past the 65535
  // per-dimension ceiling; sdpa_softmax.wgsl recovers the flat row index from
  // @builtin(num_workgroups), so no override constant is needed here.
  utils::WgCount softmax_grid = utils::compute_2d_workgroup_count(
      device,
      static_cast<uint32_t>(num_rows),
      /*workgroup_size=*/1,
      "et_vk_sdpa_softmax");
  utils::check_fp32(out, "et_vk.sdpa", "out");

  // NOTE: graph.create_scratch_buffer allocates a fresh buffer per call (no
  // pooling), so a multi-layer graph holds 2 × num_layers × aw_bytes live, not
  // 2 × max(aw_bytes) (e.g. S=1024 × 12 layers ≈ 1.1 GB) — a memory-headroom
  // gap in the shared allocator, not this op; not a correctness issue.
  WGPUBuffer attn_buf = graph.create_scratch_buffer(aw_bytes);
  WGPUBuffer softmax_buf = graph.create_scratch_buffer(aw_bytes);

  // ---- Dispatch 1: QK (per-row for standard attn, per-entry for channel) ----
  {
    QkParams p = {};
    p.B = B;
    p.H = H;
    p.S_q = S_q;
    p.S_kv = S_kv;
    p.D = D;
    p.has_mask = has_mask ? 1u : 0u;
    p.scale = scale;
    WGPUBuffer uniform_buffer =
        utils::make_uniform(device, &p, sizeof(QkParams));
    graph.add_uniform_buffer_bytes(sizeof(QkParams));

    // 4-byte dummy storage to satisfy the mask binding when absent (shader
    // never reads it under has_mask == 0).
    utils::OptionalBinding mask = utils::make_optional_binding(
        device,
        has_mask,
        has_mask ? graph.get_tensor(mask_id).buffer : nullptr,
        has_mask ? graph.get_tensor(mask_id).nbytes : 0);

    WGPUConstantEntry wg_const = utils::make_wg_size_constant(qk_wg_size);

    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        qk_per_entry ? kEtVkSdpaQkEntryWGSL : kEtVkSdpaQkWGSL,
        {
            {0, WGPUBufferBindingType_Storage, attn_buf, aw_bytes},
            {1, WGPUBufferBindingType_ReadOnlyStorage, q.buffer, q.nbytes},
            {2, WGPUBufferBindingType_ReadOnlyStorage, k.buffer, k.nbytes},
            {3,
             WGPUBufferBindingType_ReadOnlyStorage,
             mask.buffer,
             mask.nbytes},
            {4,
             WGPUBufferBindingType_Uniform,
             uniform_buffer,
             sizeof(QkParams)},
        },
        &wg_const,
        1);

    if (qk_per_entry) {
      graph.add_dispatch_2d(
          bundle.pipeline, bundle.bind_group, qk_entry_grid.x, qk_entry_grid.y);
    } else {
      graph.add_dispatch({bundle.pipeline, bundle.bind_group, qk_wg_count});
    }

    wgpuBufferRelease(uniform_buffer);
    if (mask.owned_dummy != nullptr) {
      wgpuBufferRelease(mask.owned_dummy);
    }
  }

  // ---- Dispatch 2: softmax over the last dim (reuse sdpa_softmax.wgsl) ----
  {
    SoftmaxParams p = {};
    p.num_rows = static_cast<uint32_t>(num_rows);
    p.row_width = S_kv;
    WGPUBuffer uniform_buffer =
        utils::make_uniform(device, &p, sizeof(SoftmaxParams));
    graph.add_uniform_buffer_bytes(sizeof(SoftmaxParams));

    // sdpa_softmax.wgsl hardcodes @workgroup_size(64,1,1); no override
    // constant is needed to decode the near-square 2D grid.
    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kSdpaSoftmaxWGSL,
        {
            {0, WGPUBufferBindingType_Storage, softmax_buf, aw_bytes},
            {1, WGPUBufferBindingType_ReadOnlyStorage, attn_buf, aw_bytes},
            {2,
             WGPUBufferBindingType_Uniform,
             uniform_buffer,
             sizeof(SoftmaxParams)},
        });

    graph.add_dispatch_2d(
        bundle.pipeline, bundle.bind_group, softmax_grid.x, softmax_grid.y);

    wgpuBufferRelease(uniform_buffer);
  }

  // ---- Dispatch 3: AV (one thread per (b,h,s,d4) vec4 output element) ----
  {
    AvParams p = {};
    p.B = B;
    p.H = H;
    p.S_q = S_q;
    p.S_kv = S_kv;
    p.D = D;
    WGPUBuffer uniform_buffer =
        utils::make_uniform(device, &p, sizeof(AvParams));
    graph.add_uniform_buffer_bytes(sizeof(AvParams));

    WGPUConstantEntry wg_const = utils::make_wg_size_constant(av_wg_size);

    utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
        device,
        kEtVkSdpaAvWGSL,
        {
            {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
            {1, WGPUBufferBindingType_ReadOnlyStorage, softmax_buf, aw_bytes},
            {2, WGPUBufferBindingType_ReadOnlyStorage, v.buffer, v.nbytes},
            {3,
             WGPUBufferBindingType_Uniform,
             uniform_buffer,
             sizeof(AvParams)},
        },
        &wg_const,
        1);

    graph.add_dispatch({bundle.pipeline, bundle.bind_group, av_wg_count});

    wgpuBufferRelease(uniform_buffer);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.sdpa.default, et_vk_sdpa_impl);
}

} // namespace executorch::backends::webgpu

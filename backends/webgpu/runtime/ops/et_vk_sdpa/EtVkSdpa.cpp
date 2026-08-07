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
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_qk_entry_exact_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_qk_entry_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/et_vk_sdpa/et_vk_sdpa_qk_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa/sdpa_softmax_wgsl.h>

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

enum class SdpaLayout : uint32_t {
  BHSD = 0,
  BSHD = 1,
};

enum class MaskMode : uint32_t {
  None = 0,
  Rank2 = 1,
  Expanded = 2,
};

struct QkParams {
  uint32_t B;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t S_q;
  uint32_t S_kv;
  uint32_t D;
  uint32_t g;
  uint32_t has_mask;
  uint32_t mask_mode;
  uint32_t layout;
  uint32_t _pad0;
  float scale;
  // This dispatch owns batch-head pairs [bh_lo, bh_lo + bh_count). The scratch
  // is indexed relative to bh_lo; q/k/v stay absolute.
  uint32_t bh_lo;
  uint32_t bh_count;
  uint32_t elide_masked_qk;
  uint32_t _pad2;
};
static_assert(sizeof(QkParams) == 64, "QkParams must be 64 bytes");

struct AvParams {
  uint32_t B;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t S_q;
  uint32_t S_kv;
  uint32_t D;
  uint32_t g;
  uint32_t layout;
  uint32_t bh_lo;
  uint32_t bh_count;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(AvParams) == 48, "AvParams must be 48 bytes");

struct SoftmaxParams {
  uint32_t num_rows;
  uint32_t row_width;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(SoftmaxParams) == 16, "SoftmaxParams must be 16 bytes");

struct TensorShape {
  uint32_t B;
  uint32_t H;
  uint32_t S;
  uint32_t D;
};

struct SdpaShape {
  uint32_t B;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t S_q;
  uint32_t S_kv;
  uint32_t D;
  uint32_t g;
};

// One recorded dispatch set owns a batch-head range, its uniforms, and either
// one fixed QK dispatch or a dynamic row/entry route followed by softmax + AV.
struct ChunkRecord {
  uint32_t bh_lo;
  uint32_t bh_count;
  WGPUBuffer qk_params;
  WGPUBuffer softmax_params;
  WGPUBuffer av_params;
  size_t qk_dispatch;
  size_t qk_route_group;
  bool fixed_qk_entry;
  bool dual_qk;
  size_t softmax_dispatch;
  size_t av_dispatch;
};

struct LiveState {
  QkParams qk;
  AvParams av;
  SoftmaxParams softmax;
  utils::WgCount qk_row_grid;
  utils::WgCount qk_entry_grid;
  utils::WgCount softmax_grid;
  utils::WgCount av_grid;
  bool use_qk_entry;
};

constexpr uint32_t kQkWorkgroupSize = 64;
constexpr uint32_t kAvWorkgroupSize = 64;
constexpr uint32_t kQkTileM = 8;
constexpr uint32_t kQkTileN = 4;
// One size serves both QK kernels; pin it so a .wgsl retune cannot drift.
static_assert(
    kQkWorkgroupSize == kEtVkSdpaQkWorkgroupSizeX &&
        kQkWorkgroupSize == kEtVkSdpaQkEntryWorkgroupSizeX &&
        kQkWorkgroupSize == kEtVkSdpaQkEntryExactWorkgroupSizeX,
    "QK host workgroup size must match generated QK shader constants");
static_assert(
    kAvWorkgroupSize == kEtVkSdpaAvWorkgroupSizeX,
    "AV host workgroup size must match the generated AV shader constant");
// Below this occupancy the per-entry QK kernel beats the per-row kernel.
constexpr uint32_t kQkEntryOccupancyFloor = 4096;

uint32_t checked_u32(uint64_t value, const char* label) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(
        std::string("WebGPU SDPA: ") + label + " exceeds uint32");
  }
  return static_cast<uint32_t>(value);
}

uint64_t checked_mul(uint64_t lhs, uint64_t rhs, const char* label) {
  if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    throw std::runtime_error(
        std::string("WebGPU SDPA: ") + label + " overflow");
  }
  return lhs * rhs;
}

uint64_t numel(const std::vector<int64_t>& dims, const char* label) {
  uint64_t value = 1;
  for (int64_t dim : dims) {
    if (dim <= 0) {
      throw std::runtime_error(
          std::string("WebGPU SDPA: non-positive ") + label + " dimension");
    }
    value = checked_mul(value, static_cast<uint64_t>(dim), label);
  }
  return value;
}

TensorShape parse_shape(
    const std::vector<int64_t>& dims,
    SdpaLayout layout,
    const char* label) {
  if (dims.size() < 3) {
    throw std::runtime_error(
        std::string("WebGPU SDPA: ") + label + " rank must be at least 3");
  }
  const size_t rank = dims.size();
  const size_t h_dim = layout == SdpaLayout::BHSD ? rank - 3 : rank - 2;
  const size_t s_dim = layout == SdpaLayout::BHSD ? rank - 2 : rank - 3;
  uint64_t batch = 1;
  for (size_t i = 0; i + 3 < rank; ++i) {
    if (dims[i] <= 0) {
      throw std::runtime_error("WebGPU SDPA: non-positive batch dimension");
    }
    batch = checked_mul(batch, static_cast<uint64_t>(dims[i]), "batch");
  }
  return {
      checked_u32(batch, "batch"),
      checked_u32(static_cast<uint64_t>(dims[h_dim]), "heads"),
      checked_u32(static_cast<uint64_t>(dims[s_dim]), "sequence"),
      checked_u32(static_cast<uint64_t>(dims[rank - 1]), "head dimension")};
}

void check_fp32(const WebGPUTensor& tensor, const char* label) {
  const uint64_t expected =
      checked_mul(numel(tensor.dims, label), sizeof(float), label);
  if (tensor.elem_size != sizeof(float) || tensor.is_int ||
      tensor.nbytes != expected) {
    throw std::runtime_error(
        std::string("WebGPU SDPA: ") + label + " must be fp32");
  }
}

SdpaShape validate_shapes(
    const std::vector<int64_t>& q_dims,
    const std::vector<int64_t>& k_dims,
    const std::vector<int64_t>& v_dims,
    SdpaLayout layout) {
  if (q_dims.size() != k_dims.size() || k_dims.size() != v_dims.size()) {
    throw std::runtime_error("WebGPU SDPA: q/k/v rank mismatch");
  }
  const TensorShape q = parse_shape(q_dims, layout, "q");
  const TensorShape k = parse_shape(k_dims, layout, "k");
  const TensorShape v = parse_shape(v_dims, layout, "v");
  if (q.B != k.B || k.B != v.B) {
    throw std::runtime_error("WebGPU SDPA: q/k/v batch mismatch");
  }
  if (q.D != k.D || k.D != v.D) {
    throw std::runtime_error("WebGPU SDPA: q/k/v head dimension mismatch");
  }
  if (k.H != v.H || k.S != v.S) {
    throw std::runtime_error("WebGPU SDPA: k/v shape mismatch");
  }
  if (q.H % k.H != 0) {
    throw std::runtime_error("WebGPU SDPA: Hq must be divisible by Hkv");
  }
  if (q.D % 4 != 0) {
    throw std::runtime_error(
        "WebGPU SDPA: head dimension must be a multiple of 4");
  }
  return {q.B, q.H, k.H, q.S, k.S, q.D, q.H / k.H};
}

MaskMode validate_mask(
    WebGPUGraph& graph,
    int mask_id,
    const SdpaShape& shape,
    bool require_rank2,
    bool live) {
  using VT = WebGPUGraph::ValueType;
  const VT type = graph.get_value_type(mask_id);
  if (type == VT::Null) {
    if (require_rank2) {
      throw std::runtime_error(
          "WebGPU et_vk.gemma4_sdpa requires an additive attn_mask");
    }
    return MaskMode::None;
  }
  if (type != VT::Tensor) {
    throw std::runtime_error("WebGPU SDPA: attn_mask must be a tensor or None");
  }
  const auto& mask = graph.get_tensor(mask_id);
  if (!live) {
    check_fp32(mask, "attn_mask");
  }
  const auto& dims = live ? graph.cur_dims(mask_id) : mask.dims;
  if (dims.size() == 2 && dims[0] == shape.S_q && dims[1] == shape.S_kv) {
    return MaskMode::Rank2;
  }
  if (!require_rank2 && dims.size() == 4 && dims[0] == shape.B &&
      dims[1] == shape.Hq && dims[2] == shape.S_q && dims[3] == shape.S_kv) {
    return MaskMode::Expanded;
  }
  if (require_rank2) {
    std::string actual = "[";
    for (size_t i = 0; i < dims.size(); i++) {
      actual += (i == 0 ? "" : ",") + std::to_string(dims[i]);
    }
    actual += "]";
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: attn_mask " + actual +
        " must equal [S_q,S_kv]=[" + std::to_string(shape.S_q) + "," +
        std::to_string(shape.S_kv) + "]");
  }
  throw std::runtime_error(
      "WebGPU et_vk.sdpa: attn_mask must be [S_q, S_kv] or "
      "[B, Hq, S_q, S_kv]");
}

LiveState make_live_state(
    WGPUDevice device,
    const SdpaShape& shape,
    MaskMode mask_mode,
    SdpaLayout layout,
    float scale,
    uint32_t qk_wg,
    uint32_t av_wg,
    uint32_t bh_lo,
    uint32_t bh_count,
    bool allow_masked_qk_elision) {
  // Grids and the scratch extent cover only this chunk's batch-head pairs.
  const uint64_t num_rows64 =
      checked_mul(bh_count, shape.S_q, "attention rows");
  const uint64_t aw_numel64 =
      checked_mul(num_rows64, shape.S_kv, "attention elements");
  const uint64_t out_vec4_64 =
      checked_mul(num_rows64, shape.D / 4u, "output vec4 elements");
  const uint64_t qk_tiles64 = checked_mul(
      bh_count,
      checked_mul(
          utils::div_up(shape.S_q, kQkTileM),
          utils::div_up(shape.S_kv, kQkTileN),
          "attention qk tiles"),
      "attention qk tiles");
  const uint32_t num_rows = checked_u32(num_rows64, "attention rows");
  const uint32_t aw_numel = checked_u32(aw_numel64, "attention elements");
  const uint32_t out_vec4 = checked_u32(out_vec4_64, "output vec4 elements");
  const uint32_t qk_tiles = checked_u32(qk_tiles64, "attention qk tiles");

  LiveState state = {};
  state.qk = {
      shape.B,
      shape.Hq,
      shape.Hkv,
      shape.S_q,
      shape.S_kv,
      shape.D,
      shape.g,
      mask_mode == MaskMode::None ? 0u : 1u,
      static_cast<uint32_t>(mask_mode),
      static_cast<uint32_t>(layout),
      0u,
      scale};
  state.av = {
      shape.B,
      shape.Hq,
      shape.Hkv,
      shape.S_q,
      shape.S_kv,
      shape.D,
      shape.g,
      static_cast<uint32_t>(layout)};
  state.qk.bh_lo = bh_lo;
  state.qk.bh_count = bh_count;
  state.qk.elide_masked_qk = allow_masked_qk_elision &&
          mask_mode == MaskMode::Rank2 && layout == SdpaLayout::BSHD &&
          scale == 1.0f && shape.B == 1u && shape.S_q == 1u && shape.Hq == 8u &&
          shape.Hkv == 1u && shape.g == 8u &&
          (shape.D == 256u || shape.D == 512u) && shape.S_kv <= 4096u
      ? 1u
      : 0u;
  state.av.bh_lo = bh_lo;
  state.av.bh_count = bh_count;
  state.softmax = {num_rows, shape.S_kv, 0u, 0u};
  state.qk_row_grid = utils::compute_2d_workgroup_count(
      device, qk_tiles, qk_wg, "et_vk_sdpa_qk");
  state.qk_entry_grid = utils::compute_2d_workgroup_count(
      device, aw_numel, qk_wg, "et_vk_sdpa_qk_entry");
  state.softmax_grid = utils::compute_2d_workgroup_count(
      device, num_rows, 1u, "et_vk_sdpa_softmax");
  state.av_grid = utils::compute_2d_workgroup_count(
      device, out_vec4, av_wg, "et_vk_sdpa_av");
  state.use_qk_entry = num_rows < kQkEntryOccupancyFloor;
  return state;
}

WGPUConstantEntry workgroup_constant(uint32_t size) {
  WGPUConstantEntry entry = {};
  entry.key = {"wg_size", WGPU_STRLEN};
  entry.value = static_cast<double>(size);
  return entry;
}

size_t record_qk_dispatch(
    WebGPUGraph& graph,
    const char* shader,
    const char* label,
    const WebGPUTensor& q,
    const WebGPUTensor& k,
    WGPUBuffer mask_buffer,
    uint64_t mask_nbytes,
    WGPUBuffer attn_buffer,
    uint64_t aw_bytes,
    uint64_t attn_offset,
    WGPUBuffer params_buffer,
    uint32_t qk_wg,
    utils::WgCount grid) {
  const WGPUConstantEntry constant = workgroup_constant(qk_wg);
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      graph.device(),
      shader,
      {
          {0,
           WGPUBufferBindingType_Storage,
           attn_buffer,
           aw_bytes,
           attn_offset},
          {1, WGPUBufferBindingType_ReadOnlyStorage, q.buffer, q.nbytes},
          {2, WGPUBufferBindingType_ReadOnlyStorage, k.buffer, k.nbytes},
          {3, WGPUBufferBindingType_ReadOnlyStorage, mask_buffer, mask_nbytes},
          {4, WGPUBufferBindingType_Uniform, params_buffer, sizeof(QkParams)},
      },
      &constant,
      1);
  return graph.add_dispatch(
      {bundle.pipeline, bundle.bind_group, grid.x, label, grid.y});
}

void rewrite_live_state(
    WebGPUGraph& graph,
    const LiveState& state,
    WGPUBuffer qk_params,
    WGPUBuffer softmax_params,
    WGPUBuffer av_params,
    size_t qk_dispatch,
    bool fixed_qk_entry,
    bool dual_qk,
    size_t qk_route_group,
    size_t softmax_dispatch,
    size_t av_dispatch) {
  wgpuQueueWriteBuffer(
      graph.queue(), qk_params, 0, &state.qk, sizeof(state.qk));
  wgpuQueueWriteBuffer(
      graph.queue(), softmax_params, 0, &state.softmax, sizeof(state.softmax));
  wgpuQueueWriteBuffer(
      graph.queue(), av_params, 0, &state.av, sizeof(state.av));

  if (dual_qk) {
    graph.select_dispatch_route(
        qk_route_group,
        state.use_qk_entry ? 1u : 0u,
        {state.use_qk_entry ? state.qk_entry_grid : state.qk_row_grid});
  } else {
    const utils::WgCount grid =
        fixed_qk_entry ? state.qk_entry_grid : state.qk_row_grid;
    graph.dispatch_at(qk_dispatch).workgroup_count_x = grid.x;
    graph.dispatch_at(qk_dispatch).workgroup_count_y = grid.y;
  }
  graph.dispatch_at(softmax_dispatch).workgroup_count_x = state.softmax_grid.x;
  graph.dispatch_at(softmax_dispatch).workgroup_count_y = state.softmax_grid.y;
  graph.dispatch_at(av_dispatch).workgroup_count_x = state.av_grid.x;
  graph.dispatch_at(av_dispatch).workgroup_count_y = state.av_grid.y;
}

void build_sdpa(
    WebGPUGraph& graph,
    int q_id,
    int k_id,
    int v_id,
    int mask_id,
    int out_id,
    SdpaLayout layout,
    bool require_rank2_mask,
    float scale,
    bool allow_masked_qk_elision) {
  const auto& q = graph.get_tensor(q_id);
  const auto& k = graph.get_tensor(k_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& out = graph.get_tensor(out_id);
  check_fp32(q, "q");
  check_fp32(k, "k");
  check_fp32(v, "v");
  check_fp32(out, "out");
  if (out.dims != q.dims) {
    throw std::runtime_error("WebGPU SDPA: output shape must match q");
  }

  const SdpaShape max_shape = validate_shapes(q.dims, k.dims, v.dims, layout);
  const MaskMode mask_mode =
      validate_mask(graph, mask_id, max_shape, require_rank2_mask, false);
  const uint64_t aw_numel = checked_mul(
      checked_mul(
          checked_mul(max_shape.B, max_shape.Hq, "attention elements"),
          max_shape.S_q,
          "attention elements"),
      max_shape.S_kv,
      "attention elements");
  checked_u32(aw_numel, "attention elements");
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(graph.device(), &limits) != WGPUStatus_Success ||
      limits.maxStorageBufferBindingSize == 0 || limits.maxBufferSize == 0) {
    throw std::runtime_error("WebGPU SDPA: device limits unavailable");
  }
  // Report the shape that produced the size: the byte count alone is ambiguous
  // (it factors several ways) and guessing which tensor it is has already
  // produced wrong fixes on this backend.
  auto scratch_shape = [&]() {
    return " (B=" + std::to_string(max_shape.B) +
        " Hq=" + std::to_string(max_shape.Hq) +
        " S_q=" + std::to_string(max_shape.S_q) +
        " S_kv=" + std::to_string(max_shape.S_kv) + " fp32, x2 buffers)";
  };
  // maxStorageBufferBindingSize caps a binding VIEW, not the buffer. The
  // scratch is [B][Hq][S_q][S_kv], so a range of batch-head pairs is a
  // contiguous slice; bind one such chunk per dispatch set.
  const uint64_t bh_bytes =
      static_cast<uint64_t>(max_shape.S_q) * max_shape.S_kv * sizeof(float);
  const utils::RowChunking chunking = utils::compute_row_chunking(
      limits.maxStorageBufferBindingSize,
      bh_bytes,
      static_cast<uint64_t>(max_shape.B) * max_shape.Hq,
      "et_vk_sdpa attention scratch");

  const uint32_t qk_wg =
      utils::clamp_workgroup_size(graph.device(), kQkWorkgroupSize);
  const uint32_t av_wg =
      utils::clamp_workgroup_size(graph.device(), kAvWorkgroupSize);

  // Chunks run back to back in one compute pass (QK -> softmax -> AV per
  // chunk), and WebGPU orders dispatches within a pass, so one chunk's worth of
  // scratch is enough for all of them -- allocate that, not the whole B*Hq
  // extent.
  const uint64_t chunk_bytes = chunking.rows_per_chunk * bh_bytes;
  if (chunk_bytes > limits.maxBufferSize ||
      chunk_bytes > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(
        "WebGPU SDPA: chunked attention scratch is " +
        std::to_string(chunk_bytes) +
        " bytes, over the per-buffer allocation limit of " +
        std::to_string(
            std::min<uint64_t>(
                limits.maxBufferSize, std::numeric_limits<size_t>::max())) +
        scratch_shape());
  }
  WGPUBuffer attn_buffer =
      graph.acquire_scratch(static_cast<size_t>(chunk_bytes));
  WebGPUGraph::ScopedScratch attn_guard(&graph, attn_buffer);
  WGPUBuffer softmax_buffer =
      graph.acquire_scratch(static_cast<size_t>(chunk_bytes));
  WebGPUGraph::ScopedScratch softmax_guard(&graph, softmax_buffer);

  const bool has_mask = mask_mode != MaskMode::None;
  WGPUBuffer mask_buffer = has_mask ? graph.get_tensor(mask_id).buffer
                                    : graph.create_scratch_buffer(16);
  const uint64_t mask_nbytes =
      has_mask ? graph.get_tensor(mask_id).nbytes : 16u;

  const bool directly_dynamic_qk = graph.tensor_has_dynamic_dims(q_id) ||
      graph.tensor_has_dynamic_dims(k_id) ||
      graph.tensor_has_dynamic_dims(v_id) ||
      (has_mask && graph.tensor_has_dynamic_dims(mask_id));
  const bool unsafe_masked_qk_alias = has_mask &&
      (q.buffer == k.buffer || q.buffer == v.buffer || q.buffer == out.buffer ||
       q.buffer == graph.get_tensor(mask_id).buffer || k.buffer == v.buffer ||
       k.buffer == out.buffer || k.buffer == graph.get_tensor(mask_id).buffer ||
       v.buffer == out.buffer || v.buffer == graph.get_tensor(mask_id).buffer ||
       out.buffer == graph.get_tensor(mask_id).buffer);
  const bool masked_qk_elision_enabled =
      allow_masked_qk_elision && has_mask && !unsafe_masked_qk_alias;

  const uint64_t bh_total = static_cast<uint64_t>(max_shape.B) * max_shape.Hq;
  std::vector<ChunkRecord> chunks;
  for (uint32_t c = 0; c < chunking.num_chunks; c++) {
    const uint64_t bh_lo = static_cast<uint64_t>(c) * chunking.rows_per_chunk;
    const uint64_t bh_n = std::min(chunking.rows_per_chunk, bh_total - bh_lo);
    const uint64_t off = 0; // shared one-chunk scratch, reused per chunk
    const uint64_t span = bh_n * bh_bytes;
    const LiveState st = make_live_state(
        graph.device(),
        max_shape,
        mask_mode,
        layout,
        scale,
        qk_wg,
        av_wg,
        static_cast<uint32_t>(bh_lo),
        static_cast<uint32_t>(bh_n),
        masked_qk_elision_enabled);

    ChunkRecord r = {};
    r.bh_lo = static_cast<uint32_t>(bh_lo);
    r.bh_count = static_cast<uint32_t>(bh_n);
    r.qk_params = graph.make_uniform_buffer(&st.qk, sizeof(st.qk));
    r.softmax_params =
        graph.make_uniform_buffer(&st.softmax, sizeof(st.softmax));
    r.av_params = graph.make_uniform_buffer(&st.av, sizeof(st.av));
    graph.own_uniform_buffer(r.qk_params);
    graph.own_uniform_buffer(r.softmax_params);
    graph.own_uniform_buffer(r.av_params);

    const bool exact_live_entry_route =
        graph.has_dynamic_shapes() && !directly_dynamic_qk && !st.use_qk_entry;
    r.dual_qk = directly_dynamic_qk || exact_live_entry_route;
    r.fixed_qk_entry = st.use_qk_entry;
    r.qk_dispatch = 0;
    r.qk_route_group = 0;
    if (r.dual_qk) {
      const size_t row_dispatch = record_qk_dispatch(
          graph,
          kEtVkSdpaQkWGSL,
          "et_vk_sdpa_qk",
          q,
          k,
          mask_buffer,
          mask_nbytes,
          attn_buffer,
          span,
          off,
          r.qk_params,
          qk_wg,
          st.qk_row_grid);
      const size_t entry_dispatch = record_qk_dispatch(
          graph,
          exact_live_entry_route ? kEtVkSdpaQkEntryExactWGSL
                                 : kEtVkSdpaQkEntryWGSL,
          exact_live_entry_route ? "et_vk_sdpa_qk_entry_exact"
                                 : "et_vk_sdpa_qk_entry",
          q,
          k,
          mask_buffer,
          mask_nbytes,
          attn_buffer,
          span,
          off,
          r.qk_params,
          qk_wg,
          st.qk_entry_grid);
      r.qk_route_group = graph.register_dispatch_route_group(
          {{row_dispatch, row_dispatch + 1},
           {entry_dispatch, entry_dispatch + 1}});
      graph.select_dispatch_route(
          r.qk_route_group,
          st.use_qk_entry ? 1u : 0u,
          {st.use_qk_entry ? st.qk_entry_grid : st.qk_row_grid});
    } else {
      r.qk_dispatch = record_qk_dispatch(
          graph,
          st.use_qk_entry ? kEtVkSdpaQkEntryWGSL : kEtVkSdpaQkWGSL,
          st.use_qk_entry ? "et_vk_sdpa_qk_entry" : "et_vk_sdpa_qk",
          q,
          k,
          mask_buffer,
          mask_nbytes,
          attn_buffer,
          span,
          off,
          r.qk_params,
          qk_wg,
          st.use_qk_entry ? st.qk_entry_grid : st.qk_row_grid);
    }

    utils::ComputePipelineBundle sm_pipeline = utils::make_compute_pipeline(
        graph.device(),
        kSdpaSoftmaxWGSL,
        {
            {0, WGPUBufferBindingType_Storage, softmax_buffer, span, off},
            {1, WGPUBufferBindingType_ReadOnlyStorage, attn_buffer, span, off},
            {2,
             WGPUBufferBindingType_Uniform,
             r.softmax_params,
             sizeof(SoftmaxParams)},
        });
    r.softmax_dispatch = graph.add_dispatch(
        {sm_pipeline.pipeline,
         sm_pipeline.bind_group,
         st.softmax_grid.x,
         "et_vk_sdpa_softmax",
         st.softmax_grid.y});

    const WGPUConstantEntry av_constant = workgroup_constant(av_wg);
    utils::ComputePipelineBundle av_pipeline = utils::make_compute_pipeline(
        graph.device(),
        kEtVkSdpaAvWGSL,
        {
            {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
            {1,
             WGPUBufferBindingType_ReadOnlyStorage,
             softmax_buffer,
             span,
             off},
            {2, WGPUBufferBindingType_ReadOnlyStorage, v.buffer, v.nbytes},
            {3, WGPUBufferBindingType_Uniform, r.av_params, sizeof(AvParams)},
        },
        &av_constant,
        1);
    r.av_dispatch = graph.add_dispatch(
        {av_pipeline.pipeline,
         av_pipeline.bind_group,
         st.av_grid.x,
         "et_vk_sdpa_av",
         st.av_grid.y});
    chunks.push_back(r);
  }

  auto resize = [q_id,
                 k_id,
                 v_id,
                 mask_id,
                 out_id,
                 layout,
                 require_rank2_mask,
                 max_shape,
                 scale,
                 qk_wg,
                 av_wg,
                 masked_qk_elision_enabled,
                 chunks](WebGPUGraph& gr) {
    const SdpaShape live_shape = validate_shapes(
        gr.cur_dims(q_id), gr.cur_dims(k_id), gr.cur_dims(v_id), layout);
    if (live_shape.B != max_shape.B || live_shape.Hq != max_shape.Hq ||
        live_shape.Hkv != max_shape.Hkv || live_shape.D != max_shape.D ||
        live_shape.S_q > max_shape.S_q || live_shape.S_kv > max_shape.S_kv) {
      throw std::runtime_error(
          "WebGPU SDPA: live shape exceeds allocation bounds");
    }
    const MaskMode live_mask =
        validate_mask(gr, mask_id, live_shape, require_rank2_mask, true);
    // The batch-head split is shape-independent (B and Hq cannot change here),
    // so each chunk keeps its range and only its grids/params are rewritten.
    for (const ChunkRecord& r : chunks) {
      const LiveState state = make_live_state(
          gr.device(),
          live_shape,
          live_mask,
          layout,
          scale,
          qk_wg,
          av_wg,
          r.bh_lo,
          r.bh_count,
          masked_qk_elision_enabled);
      rewrite_live_state(
          gr,
          state,
          r.qk_params,
          r.softmax_params,
          r.av_params,
          r.qk_dispatch,
          r.fixed_qk_entry,
          r.dual_qk,
          r.qk_route_group,
          r.softmax_dispatch,
          r.av_dispatch);
    }
    gr.set_cur_dims(out_id, gr.cur_dims(q_id));
  };
  graph.add_tensor_resize_hook(q_id, resize);
  graph.add_tensor_resize_hook(k_id, resize);
  graph.add_tensor_resize_hook(v_id, resize);
  if (has_mask) {
    graph.add_tensor_resize_hook(mask_id, resize);
  }
}

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

  const auto& q = graph.get_tensor(q_id);
  const TensorShape q_shape = parse_shape(q.dims, SdpaLayout::BHSD, "q");
  float scale = 1.0f / std::sqrt(static_cast<float>(q_shape.D));
  const auto scale_type = graph.get_value_type(scale_id);
  if (scale_type == WebGPUGraph::ValueType::Double) {
    scale = static_cast<float>(graph.get_double(scale_id));
  } else if (scale_type != WebGPUGraph::ValueType::Null) {
    throw std::runtime_error("WebGPU et_vk.sdpa: scale must be Double or None");
  }
  if (!std::isfinite(scale)) {
    throw std::runtime_error("WebGPU et_vk.sdpa: scale must be finite");
  }
  build_sdpa(
      graph,
      q_id,
      k_id,
      v_id,
      mask_id,
      out_id,
      SdpaLayout::BHSD,
      false,
      scale,
      false);
}

void gemma4_sdpa_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // Fence: only the Gemma4 exporter's masked, non-causal, scale=1.0 ABI.
  if (args.size() != 9) {
    throw std::runtime_error("WebGPU et_vk.gemma4_sdpa: expected 9 args");
  }
  const int q_id = args.at(0);
  const int k_id = args.at(1);
  const int v_id = args.at(2);
  const int start_id = args.at(3);
  const int mask_id = args.at(4);
  const int dropout_id = args.at(5);
  const int causal_id = args.at(6);
  const int scale_id = args.at(7);
  const int out_id = args.at(8);
  using VT = WebGPUGraph::ValueType;

  const VT start_type = graph.get_value_type(start_id);
  int64_t start = 0;
  if (start_type == VT::Int) {
    start = graph.get_int(start_id);
  } else if (start_type == VT::SymInt) {
    start = graph.read_symint(start_id);
  } else {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: start_pos must be Int or SymInt");
  }
  if (start < 0) {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: start_pos must be non-negative");
  }

  const VT dropout_type = graph.get_value_type(dropout_id);
  const bool dropout_zero = dropout_type == VT::Null ||
      (dropout_type == VT::Double && graph.get_double(dropout_id) == 0.0) ||
      (dropout_type == VT::Int && graph.get_int(dropout_id) == 0);
  if (!dropout_zero) {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: only dropout_p=0 is supported");
  }
  const VT causal_type = graph.get_value_type(causal_id);
  const bool causal_false = causal_type == VT::Null ||
      (causal_type == VT::Bool && !graph.get_bool(causal_id));
  if (!causal_false) {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: only is_causal=false is supported");
  }
  if (graph.get_value_type(mask_id) != VT::Tensor) {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa requires an additive attn_mask");
  }
  const VT scale_type = graph.get_value_type(scale_id);
  // Vulkan interning can serialize semantic 1.0 as Int(1); accept both.
  double scale_value;
  if (scale_type == VT::Double) {
    scale_value = graph.get_double(scale_id);
  } else if (scale_type == VT::Int) {
    scale_value = static_cast<double>(graph.get_int(scale_id));
  } else {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: requires explicit scale=1.0");
  }
  if (!std::isfinite(scale_value)) {
    throw std::runtime_error("WebGPU et_vk.gemma4_sdpa: scale must be finite");
  }
  if (scale_value != 1.0) {
    throw std::runtime_error(
        "WebGPU et_vk.gemma4_sdpa: requires explicit scale=1.0");
  }
  const float scale = static_cast<float>(scale_value);

  build_sdpa(
      graph,
      q_id,
      k_id,
      v_id,
      mask_id,
      out_id,
      SdpaLayout::BSHD,
      true,
      scale,
      true);
  if (start_type == VT::SymInt) {
    graph.add_resize_hook(start_id, [start_id](WebGPUGraph& gr) {
      if (gr.read_symint(start_id) < 0) {
        throw std::runtime_error(
            "WebGPU et_vk.gemma4_sdpa: start_pos must be non-negative");
      }
    });
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.sdpa.default, et_vk_sdpa_impl);
  WEBGPU_REGISTER_OP(et_vk.gemma4_sdpa.default, gemma4_sdpa_impl);
}

} // namespace executorch::backends::webgpu

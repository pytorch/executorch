/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/SdpaFdDecode.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace executorch::backends::webgpu {

namespace {

// Register-tile dims; MUST match TM/TN in the reg WGSL kernels.
constexpr int64_t kSdpaTileM = 4;
constexpr int64_t kSdpaTileN = 4;

constexpr const char* kUpdateCacheShader = "update_cache";
constexpr const char* kUpdateCacheHalfShader = "update_cache_half";
constexpr const char* kAttnWeightsShader = "sdpa_compute_attn_weights";
constexpr const char* kAttnWeightsHalfShader = "sdpa_compute_attn_weights_half";
constexpr const char* kSoftmaxShader = "sdpa_softmax";
constexpr const char* kComputeOutShader = "sdpa_compute_out";
constexpr const char* kComputeOutHalfShader = "sdpa_compute_out_half";
constexpr const char* kStreamingK16Shader =
    "streaming_attention_k16_causal_bound";
constexpr const char* kStreamingQwen3K16Shader =
    "streaming_attention_qwen3_k16_causal_bound";
constexpr const char* kStreamingQwen3Q32K16Shader =
    "streaming_attention_qwen3_q32_k16_causal_bound";

// Uniform param structs (all 16-byte aligned, matching the WGSL Params).
struct UpdateCacheParams {
  uint32_t numel;
  uint32_t dst_offset;
  uint32_t cache_numel;
  uint32_t _pad0;
};
static_assert(sizeof(UpdateCacheParams) == 16, "UpdateCacheParams must be 16B");

struct AttnWeightsParams {
  uint32_t S;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t input_pos;
  uint32_t g;
  float scale;
};
static_assert(sizeof(AttnWeightsParams) == 32, "AttnWeightsParams must be 32B");

struct SoftmaxParams {
  uint32_t num_rows;
  uint32_t row_width;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(SoftmaxParams) == 16, "SoftmaxParams must be 16B");

struct ComputeOutParams {
  uint32_t S;
  uint32_t Hq;
  uint32_t Hkv;
  uint32_t D;
  uint32_t context_len;
  uint32_t g;
  uint32_t _pad0;
  uint32_t _pad1;
};
static_assert(sizeof(ComputeOutParams) == 32, "ComputeOutParams must be 32B");

struct StreamingAttentionK16Params {
  uint32_t S;
  uint32_t context_len;
  uint32_t input_pos;
  uint32_t q_token_stride4;
  uint32_t q_head_stride4;
  uint32_t kv_token_stride4;
  uint32_t kv_head_stride4;
  uint32_t o_token_stride4;
  uint32_t o_head_stride4;
  uint32_t _pad0;
  uint32_t _pad1;
  uint32_t _pad2;
};
static_assert(
    sizeof(StreamingAttentionK16Params) == 48,
    "StreamingAttentionK16Params must be 48B");

struct SdpaLiveState {
  int64_t s;
  int64_t pos;
  int64_t context_len;
  UpdateCacheParams update_cache;
  AttnWeightsParams attn_weights;
  SoftmaxParams softmax;
  ComputeOutParams compute_out;
  StreamingAttentionK16Params streaming_k16;
  utils::WgCount update_cache_grid;
  utils::WgCount qk_grid;
  utils::WgCount softmax_grid;
  utils::WgCount av_grid;
  utils::WgCount streaming_k16_grid;
  bool use_fd;
  SdpaFdDecodeState fd;
};

// Param-struct builder helpers — used in both initial build and resize hook.
static UpdateCacheParams make_update_cache_params(
    uint64_t kv_numel,
    uint32_t dst_offset,
    uint64_t cache_numel) {
  UpdateCacheParams p = {};
  p.numel = static_cast<uint32_t>(kv_numel);
  p.dst_offset = dst_offset;
  p.cache_numel = static_cast<uint32_t>(cache_numel);
  return p;
}

static AttnWeightsParams make_attn_weights_params(
    int64_t S,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t ctx,
    int64_t pos,
    int64_t g,
    float scale) {
  AttnWeightsParams p = {};
  p.S = static_cast<uint32_t>(S);
  p.Hq = static_cast<uint32_t>(Hq);
  p.Hkv = static_cast<uint32_t>(Hkv);
  p.D = static_cast<uint32_t>(D);
  p.context_len = static_cast<uint32_t>(ctx);
  p.input_pos = static_cast<uint32_t>(pos);
  p.g = static_cast<uint32_t>(g);
  p.scale = scale;
  return p;
}

static SoftmaxParams make_softmax_params(int64_t Hq, int64_t S, int64_t ctx) {
  SoftmaxParams p = {};
  p.num_rows = static_cast<uint32_t>(Hq * S);
  p.row_width = static_cast<uint32_t>(ctx);
  return p;
}

static ComputeOutParams make_compute_out_params(
    int64_t S,
    int64_t Hq,
    int64_t Hkv,
    int64_t D,
    int64_t ctx,
    int64_t g) {
  ComputeOutParams p = {};
  p.S = static_cast<uint32_t>(S);
  p.Hq = static_cast<uint32_t>(Hq);
  p.Hkv = static_cast<uint32_t>(Hkv);
  p.D = static_cast<uint32_t>(D);
  p.context_len = static_cast<uint32_t>(ctx);
  p.g = static_cast<uint32_t>(g);
  return p;
}

static StreamingAttentionK16Params make_streaming_attention_k16_params(
    int64_t S,
    int64_t context_len,
    int64_t input_pos,
    int64_t Hq,
    int64_t Hkv,
    int64_t D) {
  StreamingAttentionK16Params p = {};
  p.S = static_cast<uint32_t>(S);
  p.context_len = static_cast<uint32_t>(context_len);
  p.input_pos = static_cast<uint32_t>(input_pos);
  p.q_token_stride4 = static_cast<uint32_t>(Hq * D / 4);
  p.q_head_stride4 = static_cast<uint32_t>(D / 4);
  p.kv_token_stride4 = static_cast<uint32_t>(Hkv * D / 4);
  p.kv_head_stride4 = static_cast<uint32_t>(D / 4);
  p.o_token_stride4 = static_cast<uint32_t>(Hq * D / 4);
  p.o_head_stride4 = static_cast<uint32_t>(D / 4);
  return p;
}

static bool streaming_attention_k16_device_supported(WGPUDevice device) {
  WGPULimits limits = {};
  const WebGPUContext* context = get_default_webgpu_context();
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup >= 128u &&
      limits.maxComputeWorkgroupSizeX >= 32u &&
      limits.maxComputeWorkgroupSizeY >= 4u &&
      limits.maxComputeWorkgroupStorageSize >= 14720u;
}

constexpr uint32_t kLlamaK16QueryTile = 32u;
constexpr uint32_t kQwen3K16QueryTile = 16u;
constexpr uint32_t kQwen3Q32K16QueryTile = 32u;
constexpr uint32_t kQwen3Q16K16StorageBytes = 512u * 4u * sizeof(float) +
    512u * 4u * sizeof(uint16_t) + 128u * 2u * sizeof(float) +
    3u * 16u * sizeof(float);
constexpr uint32_t kQwen3K16StorageBytes = kQwen3Q16K16StorageBytes;
// Mirrors the Q32 shader's workgroup arrays (t_q_tile vec4<f32>x1024, t_kv_tile
// vec4<f16>x512, t_scores vec2<f32>x256, t_m/t_d/t_alpha f32x32) so the
// device-support gate stays tied to the declared storage, not a literal.
constexpr uint32_t kQwen3Q32K16StorageBytes = 1024u * 4u * sizeof(float) +
    512u * 4u * sizeof(uint16_t) + 256u * 2u * sizeof(float) +
    3u * 32u * sizeof(float);

constexpr bool streaming_attention_k16_workgroup_count_fits(
    int64_t S,
    int64_t Hkv,
    int64_t g,
    uint32_t query_tile,
    uint32_t max_workgroups) {
  if (S <= 0 || Hkv <= 0 || g <= 0 || query_tile == 0u ||
      max_workgroups == 0u) {
    return false;
  }
  if (static_cast<uint64_t>(S) > UINT64_MAX / static_cast<uint64_t>(g)) {
    return false;
  }
  const uint64_t logical_rows =
      static_cast<uint64_t>(S) * static_cast<uint64_t>(g);
  if (logical_rows > UINT64_MAX - (query_tile - 1u)) {
    return false;
  }
  const uint64_t groups_per_kv = (logical_rows + query_tile - 1u) / query_tile;
  if (groups_per_kv > UINT64_MAX / static_cast<uint64_t>(Hkv)) {
    return false;
  }
  const uint64_t workgroups = groups_per_kv * static_cast<uint64_t>(Hkv);
  return workgroups > 0u && workgroups <= UINT32_MAX &&
      workgroups <= max_workgroups;
}

static_assert(
    streaming_attention_k16_workgroup_count_fits(65528, 8, 2, 16, 65535));
static_assert(
    !streaming_attention_k16_workgroup_count_fits(65529, 8, 2, 16, 65535));

bool qwen3_q16_k16_device_supported(WGPUDevice device) {
  WGPULimits limits = {};
  const WebGPUContext* context = get_default_webgpu_context();
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeWorkgroupSizeX >= 16u &&
      limits.maxComputeWorkgroupSizeY >= 8u &&
      limits.maxComputeInvocationsPerWorkgroup >= 128u &&
      limits.maxComputeWorkgroupStorageSize >= kQwen3K16StorageBytes &&
      limits.maxStorageBuffersPerShaderStage >= 4u;
}

bool qwen3_q32_k16_device_supported(WGPUDevice device) {
  WGPULimits limits = {};
  const WebGPUContext* context = get_default_webgpu_context();
  return context != nullptr && context->shader_f16_supported &&
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeWorkgroupSizeX >= 32u &&
      limits.maxComputeWorkgroupSizeY >= 8u &&
      limits.maxComputeInvocationsPerWorkgroup >= 256u &&
      limits.maxComputeWorkgroupStorageSize >= kQwen3Q32K16StorageBytes &&
      limits.maxStorageBuffersPerShaderStage >= 4u;
}

static utils::WgCount streaming_attention_k16_grid(
    WGPUDevice device,
    int64_t S,
    int64_t Hkv,
    int64_t g,
    uint32_t query_tile) {
  const uint64_t groups_per_kv =
      (static_cast<uint64_t>(S) * static_cast<uint64_t>(g) + query_tile - 1u) /
      query_tile;
  const uint64_t workgroups = static_cast<uint64_t>(Hkv) * groups_per_kv;
  if (workgroups == 0u || workgroups > UINT32_MAX) {
    throw std::runtime_error("WebGPU sdpa: K16 workgroup count exceeds uint32");
  }
  if (workgroups > utils::queried_max_workgroups(device)) {
    throw std::runtime_error(
        "WebGPU sdpa: K16 workgroup count exceeds the 1D dispatch limit");
  }
  return {static_cast<uint32_t>(workgroups), 1u};
}

size_t add_sdpa_compute_dispatch(
    WebGPUGraph& graph,
    const char* shader_name,
    std::vector<WebGPUBufferBinding> bindings,
    WGPUBuffer uniform_buffer,
    uint64_t uniform_size,
    utils::WgCount grid,
    uint32_t wg_size,
    const char* kernel_name = "") {
  bindings.push_back({uniform_buffer, 0u, uniform_size});
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = shader_name;
  descriptor.kernel_name = kernel_name;
  descriptor.bindings = std::move(bindings);
  if (wg_size != 0) {
    descriptor.constants = {{"wg_size", static_cast<double>(wg_size)}};
  }
  descriptor.grid = {grid.x, grid.y};
  return graph.add_compute_dispatch(descriptor);
}

// Dispatch one update_cache (K or V); returns the retained uniform buffer.
static WGPUBuffer record_update_cache_dispatch(
    WebGPUGraph& graph,
    const WebGPUTensor& cache,
    const WebGPUTensor& src,
    uint64_t kv_numel,
    uint32_t kv_dst_offset,
    uint64_t cache_numel,
    uint32_t uc_wg,
    const char* label) {
  const uint32_t wgc = utils::compute_1d_workgroup_count(
      graph.device(), static_cast<uint32_t>(kv_numel), uc_wg, label);
  const UpdateCacheParams uc =
      make_update_cache_params(kv_numel, kv_dst_offset, cache_numel);
  WGPUBuffer ubuf = graph.create_params_buffer(uc);
  const std::vector<WebGPUBufferBinding> bindings = {
      {cache.buffer, 0u, cache.nbytes}, {src.buffer, 0u, src.nbytes}};
  add_sdpa_compute_dispatch(
      graph,
      graph.kv_f16() ? kUpdateCacheHalfShader : kUpdateCacheShader,
      bindings,
      ubuf,
      sizeof(uc),
      {wgc, 1u},
      uc_wg,
      "update_cache");
  return ubuf;
}

// llama.sdpa_with_kv_cache.default args mirror the Vulkan impl.
void sdpa_with_kv_cache_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int q_id = args.at(0);
  const int k_id = args.at(1);
  const int v_id = args.at(2);
  const int k_cache_id = args.at(3);
  const int v_cache_id = args.at(4);
  const int input_pos_id = args.at(5);
  // arg 6 (seq_len) is derived from q; args 7-9 validated below.
  const int attn_mask_id = args.at(7);
  const int drop_p_id = args.at(8);
  const int is_causal_id = args.at(9);
  const int scale_id = args.at(10);
  const int out_id = args.at(11);

  const auto& q = graph.get_tensor(q_id);
  const auto& k = graph.get_tensor(k_id);
  const auto& v = graph.get_tensor(v_id);
  const auto& k_cache = graph.get_tensor(k_cache_id);
  const auto& v_cache = graph.get_tensor(v_cache_id);
  const auto& out = graph.get_tensor(out_id);

  if (q.dims.size() < 3 || k.dims.size() < 3 || v.dims.size() < 3 ||
      k_cache.dims.size() < 3) {
    throw std::runtime_error("WebGPU sdpa: q/k/v/k_cache must be rank >= 3");
  }

  // q [1, S, Hq, D]; k/v [1, S, Hkv, D]; caches [1, Cmax, Hkv, D].
  const size_t qn = q.dims.size();
  const int64_t S = q.dims[qn - 3];
  const int64_t Hq = q.dims[qn - 2];
  const int64_t D = q.dims[qn - 1];

  const size_t kn = k.dims.size();
  const int64_t Hkv = k.dims[kn - 2];

  const size_t cn = k_cache.dims.size();
  const int64_t Cmax = k_cache.dims[cn - 3];

  // Validate B == 1 (leading dims must all be 1).
  for (size_t i = 0; i + 3 < qn; i++) {
    if (q.dims[i] != 1) {
      throw std::runtime_error("WebGPU sdpa: only batch size 1 is supported");
    }
  }
  if (S <= 0 || Hq <= 0 || D <= 0 || Hkv <= 0 || Cmax <= 0) {
    throw std::runtime_error("WebGPU sdpa: non-positive dimension");
  }
  if (Hq % Hkv != 0) {
    throw std::runtime_error("WebGPU sdpa: Hq must be a multiple of Hkv (GQA)");
  }
  const int64_t g = Hq / Hkv;

  // k/v seq-len must match q's S.
  if (k.dims[kn - 3] != S || v.dims[v.dims.size() - 3] != S) {
    throw std::runtime_error("WebGPU sdpa: k/v seq_len must match q");
  }

  // k/v projected shapes must match q/k; mirrors Vulkan update_cache -1/-2.
  if (k.dims[kn - 1] != D || v.dims[v.dims.size() - 1] != D) {
    throw std::runtime_error("WebGPU sdpa: k/v head_dim must match q");
  }
  // QK/AV read D as vec4 (no SDPA_PAD_D); head_dim must be a multiple of 4.
  if (D % 4 != 0) {
    throw std::runtime_error(
        "WebGPU sdpa: head_dim (D) must be a multiple of 4");
  }
  if (v.dims[v.dims.size() - 2] != Hkv) {
    throw std::runtime_error("WebGPU sdpa: v num_heads must match k");
  }

  // Mirrors Vulkan SDPA: q/k_cache head_dim + k_cache/v_cache shape must match.
  if (D != k_cache.dims[cn - 1]) {
    throw std::runtime_error("WebGPU sdpa: q and k_cache head_dim mismatch");
  }
  if (k_cache.dims[cn - 2] != Hkv) {
    throw std::runtime_error("WebGPU sdpa: k and k_cache num_heads mismatch");
  }
  if (k_cache.dims != v_cache.dims) {
    throw std::runtime_error("WebGPU sdpa: k_cache and v_cache shape mismatch");
  }

  // fp32-only: validate byte counts against fp32 element counts.
  auto numel = [](const WebGPUTensor& t) {
    uint64_t n = 1;
    for (int64_t d : t.dims) {
      n *= static_cast<uint64_t>(d);
    }
    return n;
  };
  if (q.nbytes != numel(q) * sizeof(float) ||
      k.nbytes != numel(k) * sizeof(float) ||
      v.nbytes != numel(v) * sizeof(float) ||
      out.nbytes != numel(out) * sizeof(float)) {
    throw std::runtime_error("WebGPU sdpa: fp32-only (byte-size mismatch)");
  }

  // input_pos: build-time Int (baked) OR runtime SymInt (dynamic decode).
  int64_t input_pos = 0;
  const auto input_pos_type = graph.get_value_type(input_pos_id);
  const bool dynamic_pos = input_pos_type == WebGPUGraph::ValueType::SymInt;
  if (dynamic_pos) {
    input_pos = graph.read_symint(input_pos_id); // build placeholder (e.g. 0)
  } else if (input_pos_type == WebGPUGraph::ValueType::Int) {
    input_pos = graph.get_int(input_pos_id);
  } else {
    // No silent default-to-0; mirrors Vulkan get_or_create_int_param_buffer.
    throw std::runtime_error("WebGPU sdpa: input_pos must be Int or SymInt");
  }
  if (input_pos < 0) {
    throw std::runtime_error("WebGPU sdpa: input_pos must be non-negative");
  }
  const int64_t context_len = S + input_pos;
  if (context_len <= 0 || context_len > Cmax) {
    throw std::runtime_error("WebGPU sdpa: context_len exceeds cache capacity");
  }

  // scale arg is None (use 1/sqrt(D)) or an explicit Double; reject others.
  float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const auto scale_type = graph.get_value_type(scale_id);
  if (scale_type == WebGPUGraph::ValueType::Double) {
    scale = static_cast<float>(graph.get_double(scale_id));
  } else if (scale_type != WebGPUGraph::ValueType::Null) {
    throw std::runtime_error("WebGPU sdpa: scale must be None or a Double");
  }

  // Unsupported attention args must be absent/default; mirrors Vulkan
  // SDPA.cpp:587-593 (scale is handled above as an intentional extension).
  using VT = WebGPUGraph::ValueType;
  if (graph.get_value_type(attn_mask_id) != VT::Null) {
    throw std::runtime_error("WebGPU sdpa: attn_mask is not supported");
  }
  // dropout_p: serializer may dedup 0.0 onto input_pos's Int(0) when pos=0.
  const auto drop_type = graph.get_value_type(drop_p_id);
  if (!(drop_type == VT::Null ||
        (drop_type == VT::Double && graph.get_double(drop_p_id) == 0.0) ||
        (drop_type == VT::Int && graph.get_int(drop_p_id) == 0))) {
    throw std::runtime_error("WebGPU sdpa: only dropout_p=0 is supported");
  }
  const auto causal_type = graph.get_value_type(is_causal_id);
  if (!(causal_type == VT::Null ||
        (causal_type == VT::Bool && graph.get_bool(is_causal_id)))) {
    throw std::runtime_error("WebGPU sdpa: only is_causal=true is supported");
  }

  const WGPUDevice device = graph.device();
  const WGPUBuffer k16_buffers[] = {
      q.buffer, k.buffer, v.buffer, k_cache.buffer, v_cache.buffer, out.buffer};
  bool k16_buffers_distinct = true;
  for (size_t i = 0; i < 6; i++) {
    for (size_t j = i + 1; j < 6; j++) {
      k16_buffers_distinct =
          k16_buffers_distinct && k16_buffers[i] != k16_buffers[j];
    }
  }
  // 1/sqrt(128) is not exactly representable in fp32 (unlike Llama's 0.125), so
  // match the standard scale with a tolerance to avoid a silent fallback when a
  // producer computes it via a slightly different path.
  const float qwen3_expected_scale = 1.0f / std::sqrt(128.0f);
  const bool qwen3_k16_geometry = Hq == 16 && Hkv == 8 && g == 2 && D == 128 &&
      std::fabs(scale - qwen3_expected_scale) <= 1e-6f && out.dims == q.dims;
  // Q16 is the default route for exact Qwen3 geometry; Q32 is an explicit
  // autotuning candidate requested via the sdpa_query_tile RuntimeSpec. Support
  // is evaluated per-tile so an unsupported Q32 request falls back to the Q16
  // streaming route instead of dropping to the materialized path.
  const uint32_t device_max_workgroups = utils::queried_max_workgroups(device);
  const bool qwen3_q16_supported =
      qwen3_k16_geometry && graph.kv_f16() &&
      qwen3_q16_k16_device_supported(device) &&
      streaming_attention_k16_workgroup_count_fits(
          S, Hkv, g, kQwen3K16QueryTile, device_max_workgroups);
  const bool qwen3_q32_requested = qwen3_k16_geometry &&
      graph.sdpa_query_tile() == static_cast<int>(kQwen3Q32K16QueryTile);
  const bool qwen3_q32_supported =
      qwen3_q32_requested && graph.kv_f16() &&
      qwen3_q32_k16_device_supported(device) &&
      streaming_attention_k16_workgroup_count_fits(
          S, Hkv, g, kQwen3Q32K16QueryTile, device_max_workgroups);
  const bool qwen3_q32_selected = qwen3_q32_supported;
  const bool qwen3_k16_selected = qwen3_q32_selected || qwen3_q16_supported;
  const uint32_t qwen3_query_tile =
      qwen3_q32_selected ? kQwen3Q32K16QueryTile : kQwen3K16QueryTile;
  const bool llama_k16_eligible =
      graph.kv_f16() && Hq == 32 && Hkv == 8 && g == 4 && D == 64 &&
      scale == 0.125f && out.dims == q.dims &&
      streaming_attention_k16_device_supported(device) &&
      streaming_attention_k16_workgroup_count_fits(
          S, Hkv, g, kLlamaK16QueryTile, device_max_workgroups);
  const bool k16_eligible =
      k16_buffers_distinct && (llama_k16_eligible || qwen3_k16_selected);
  const uint32_t k16_query_tile =
      qwen3_k16_selected ? qwen3_query_tile : kLlamaK16QueryTile;
  const char* k16_shader = qwen3_q32_selected ? kStreamingQwen3Q32K16Shader
      : qwen3_k16_selected                    ? kStreamingQwen3K16Shader
                                              : kStreamingK16Shader;
  const char* k16_label = qwen3_q32_selected
      ? "sdpa_streaming_attention_qwen3_q32_k16_causal_bound"
      : qwen3_k16_selected ? "sdpa_streaming_attention_qwen3_k16_causal_bound"
                           : "sdpa_streaming_attention_k16_causal_bound";
  const uint32_t uc_wg = utils::clamp_workgroup_size(
      device, get_webgpu_shader_info(kUpdateCacheShader).workgroup_size_x);
  const uint32_t qk_wg = utils::clamp_workgroup_size(
      device, get_webgpu_shader_info(kAttnWeightsShader).workgroup_size_x);
  const uint32_t av_wg = utils::clamp_workgroup_size(
      device, get_webgpu_shader_info(kComputeOutShader).workgroup_size_x);
  const uint32_t sm_wg = utils::clamp_workgroup_size_pow2(
      device, get_webgpu_shader_info(kSoftmaxShader).workgroup_size_x);
  const bool fd_eligible = D <= kSdpaFdMaxHeadDim;
  const int64_t pos_const = input_pos;

  auto compute_live_state = [q_id,
                             k_id,
                             v_id,
                             out_id,
                             qn,
                             kn,
                             S,
                             dynamic_pos,
                             input_pos_id,
                             pos_const,
                             Hq,
                             Hkv,
                             D,
                             Cmax,
                             g,
                             scale,
                             uc_wg,
                             qk_wg,
                             av_wg,
                             fd_eligible,
                             k16_eligible,
                             k16_query_tile](WebGPUGraph& gr) {
    SdpaLiveState state = {};
    const auto& q_live_dims = gr.cur_dims(q_id);
    state.s = q_live_dims[qn - 3];
    state.pos = dynamic_pos ? static_cast<int64_t>(gr.read_symint(input_pos_id))
                            : pos_const;
    if (state.s <= 0 || state.pos < 0 || state.s > S) {
      throw std::runtime_error("WebGPU sdpa: invalid live S or input_pos");
    }
    if (gr.cur_dims(k_id)[kn - 3] != state.s ||
        gr.cur_dims(v_id)[gr.cur_dims(v_id).size() - 3] != state.s) {
      throw std::runtime_error("WebGPU sdpa: live q/k/v seq_len mismatch");
    }
    const auto& out_max_dims = gr.get_tensor(out_id).dims;
    if (out_max_dims.size() != q_live_dims.size()) {
      throw std::runtime_error("WebGPU sdpa: output rank must match q");
    }
    for (size_t i = 0; i < q_live_dims.size(); i++) {
      if (q_live_dims[i] <= 0 || q_live_dims[i] > out_max_dims[i]) {
        throw std::runtime_error(
            "WebGPU sdpa: live output shape exceeds allocation");
      }
    }
    state.context_len = state.s + state.pos;
    if (state.context_len <= 0 || state.context_len > Cmax ||
        state.s > UINT32_MAX || state.pos > UINT32_MAX ||
        state.context_len > UINT32_MAX) {
      throw std::runtime_error(
          "WebGPU sdpa: live dimensions exceed cache or uint32 capacity");
    }

    const uint64_t kv_numel = static_cast<uint64_t>(state.s) *
        static_cast<uint64_t>(Hkv) * static_cast<uint64_t>(D);
    const uint64_t kv_offset = static_cast<uint64_t>(state.pos) *
        static_cast<uint64_t>(Hkv) * static_cast<uint64_t>(D);
    const uint64_t cache_numel = static_cast<uint64_t>(Cmax) *
        static_cast<uint64_t>(Hkv) * static_cast<uint64_t>(D);
    if (kv_numel > UINT32_MAX || kv_offset > UINT32_MAX ||
        cache_numel > UINT32_MAX) {
      throw std::runtime_error("WebGPU sdpa: live workload exceeds uint32");
    }

    state.update_cache = make_update_cache_params(
        kv_numel, static_cast<uint32_t>(kv_offset), cache_numel);
    state.attn_weights = make_attn_weights_params(
        state.s, Hq, Hkv, D, state.context_len, state.pos, g, scale);
    state.softmax = make_softmax_params(Hq, state.s, state.context_len);
    state.compute_out =
        make_compute_out_params(state.s, Hq, Hkv, D, state.context_len, g);
    if (k16_eligible) {
      state.streaming_k16 = make_streaming_attention_k16_params(
          state.s, state.context_len, state.pos, Hq, Hkv, D);
      state.streaming_k16_grid = streaming_attention_k16_grid(
          gr.device(), state.s, Hkv, g, k16_query_tile);
    }
    state.update_cache_grid = {
        utils::compute_1d_workgroup_count(
            gr.device(), static_cast<uint32_t>(kv_numel), uc_wg, "uc(resize)"),
        1u};
    if (!k16_eligible) {
      const uint64_t aw_floats = static_cast<uint64_t>(Hq) *
          static_cast<uint64_t>(state.s) *
          static_cast<uint64_t>(state.context_len);
      const uint64_t qk_tiles = static_cast<uint64_t>(Hq) *
          static_cast<uint64_t>(utils::div_up(state.s, kSdpaTileM)) *
          static_cast<uint64_t>(utils::div_up(state.context_len, kSdpaTileN));
      const uint64_t softmax_rows =
          static_cast<uint64_t>(Hq) * static_cast<uint64_t>(state.s);
      const uint64_t av_tiles = static_cast<uint64_t>(Hq) *
          static_cast<uint64_t>(utils::div_up(state.s, kSdpaTileM)) *
          static_cast<uint64_t>(utils::div_up(D, kSdpaTileN));
      if (aw_floats > UINT32_MAX || qk_tiles > UINT32_MAX ||
          softmax_rows > UINT32_MAX || av_tiles > UINT32_MAX) {
        throw std::runtime_error(
            "WebGPU sdpa: materialized workload exceeds uint32");
      }
      state.qk_grid = utils::compute_2d_workgroup_count(
          gr.device(), static_cast<uint32_t>(qk_tiles), qk_wg, "QK(resize)");
      state.softmax_grid = utils::compute_2d_workgroup_count(
          gr.device(),
          static_cast<uint32_t>(softmax_rows),
          1,
          "softmax(resize)");
      state.av_grid = utils::compute_2d_workgroup_count(
          gr.device(), static_cast<uint32_t>(av_tiles), av_wg, "AV(resize)");
    }
    state.use_fd = fd_eligible && state.s == 1;
    // make_sdpa_fd_decode_state requires D % 4 == 0; the op-level guard above
    // ("head_dim (D) must be a multiple of 4") rejects any other D before this
    // lambda runs, so eager construction here can never throw on it.
    if (fd_eligible) {
      state.fd = make_sdpa_fd_decode_state(
          gr.device(), Hq, Hkv, D, state.context_len, g, scale);
    }
    return state;
  };

  const SdpaLiveState initial_state = compute_live_state(graph);
  const uint64_t aw_cap_floats = k16_eligible
      ? 0u
      : static_cast<uint64_t>(Hq) * static_cast<uint64_t>(S) *
          static_cast<uint64_t>(dynamic_pos ? Cmax : context_len);
  const uint64_t aw_bytes = aw_cap_floats * sizeof(float);

  WGPUBuffer uc_k_buf = record_update_cache_dispatch(
      graph,
      k_cache,
      k,
      initial_state.update_cache.numel,
      initial_state.update_cache.dst_offset,
      initial_state.update_cache.cache_numel,
      uc_wg,
      "update_cache(K)");
  WGPUBuffer uc_v_buf = record_update_cache_dispatch(
      graph,
      v_cache,
      v,
      initial_state.update_cache.numel,
      initial_state.update_cache.dst_offset,
      initial_state.update_cache.cache_numel,
      uc_wg,
      "update_cache(V)");
  const size_t uc_k_idx = graph.num_dispatches() - 2;
  const size_t uc_v_idx = graph.num_dispatches() - 1;
  const bool dynamic_sequence = graph.tensor_has_dynamic_dims(q_id) ||
      graph.tensor_has_dynamic_dims(k_id) ||
      graph.tensor_has_dynamic_dims(v_id);
  const bool dual_route = utils::should_record_sdpa_dual_route(
      fd_eligible, dynamic_sequence, dynamic_pos);
  const bool record_k16 = k16_eligible && (dual_route || !initial_state.use_fd);
  const bool record_materialized =
      !k16_eligible && (dual_route || !initial_state.use_fd);
  const bool record_fd = dual_route || initial_state.use_fd;

  WGPUBuffer qk_buf = nullptr;
  WGPUBuffer softmax_buf = nullptr;
  WGPUBuffer av_buf = nullptr;
  size_t qk_idx = 0;
  size_t softmax_idx = 0;
  size_t av_idx = 0;
  utils::DispatchRange materialized_range = {};
  if (record_materialized) {
    WGPUBuffer attn_weights = graph.acquire_scratch(aw_bytes);
    WebGPUGraph::ScopedScratch attn_weights_guard(&graph, attn_weights);
    WGPUBuffer attn_weights_softmax = graph.acquire_scratch(aw_bytes);
    WebGPUGraph::ScopedScratch attn_weights_softmax_guard(
        &graph, attn_weights_softmax);

    materialized_range.begin = graph.num_dispatches();
    qk_buf = graph.create_params_buffer(initial_state.attn_weights);
    const std::vector<WebGPUBufferBinding> qk_bindings = {
        {attn_weights, 0u, aw_bytes},
        {q.buffer, 0u, q.nbytes},
        {k_cache.buffer, 0u, k_cache.nbytes}};
    add_sdpa_compute_dispatch(
        graph,
        graph.kv_f16() ? kAttnWeightsHalfShader : kAttnWeightsShader,
        qk_bindings,
        qk_buf,
        sizeof(AttnWeightsParams),
        initial_state.qk_grid,
        qk_wg,
        "sdpa_compute_attn_weights");
    qk_idx = graph.num_dispatches() - 1;

    softmax_buf = graph.create_params_buffer(initial_state.softmax);
    const std::vector<WebGPUBufferBinding> softmax_bindings = {
        {attn_weights_softmax, 0u, aw_bytes}, {attn_weights, 0u, aw_bytes}};
    add_sdpa_compute_dispatch(
        graph,
        kSoftmaxShader,
        softmax_bindings,
        softmax_buf,
        sizeof(SoftmaxParams),
        initial_state.softmax_grid,
        sm_wg,
        "sdpa_softmax");
    softmax_idx = graph.num_dispatches() - 1;

    av_buf = graph.create_params_buffer(initial_state.compute_out);
    const std::vector<WebGPUBufferBinding> av_bindings = {
        {out.buffer, 0u, out.nbytes},
        {attn_weights_softmax, 0u, aw_bytes},
        {v_cache.buffer, 0u, v_cache.nbytes}};
    add_sdpa_compute_dispatch(
        graph,
        graph.kv_f16() ? kComputeOutHalfShader : kComputeOutShader,
        av_bindings,
        av_buf,
        sizeof(ComputeOutParams),
        initial_state.av_grid,
        av_wg,
        "sdpa_compute_out");
    av_idx = graph.num_dispatches() - 1;
    materialized_range.end = graph.num_dispatches();
  }

  WGPUBuffer k16_buf = nullptr;
  size_t k16_idx = 0;
  utils::DispatchRange k16_range = {};
  if (record_k16) {
    k16_range.begin = graph.num_dispatches();
    k16_buf = graph.create_params_buffer(initial_state.streaming_k16);
    const std::vector<WebGPUBufferBinding> k16_bindings = {
        {out.buffer, 0u, out.nbytes},
        {q.buffer, 0u, q.nbytes},
        {k_cache.buffer, 0u, k_cache.nbytes},
        {v_cache.buffer, 0u, v_cache.nbytes}};
    const utils::WgCount initial_grid = initial_state.use_fd
        ? utils::WgCount{0u, 0u}
        : initial_state.streaming_k16_grid;
    add_sdpa_compute_dispatch(
        graph,
        k16_shader,
        k16_bindings,
        k16_buf,
        sizeof(StreamingAttentionK16Params),
        initial_grid,
        0,
        k16_label);
    k16_idx = graph.num_dispatches() - 1;
    k16_range.end = graph.num_dispatches();
  }

  SdpaFdDecodeResources fd_resources = {};
  size_t route_group = 0;
  if (record_fd) {
    fd_resources = record_sdpa_fd_decode_dispatches(
        graph, q, k_cache, v_cache, out, initial_state.fd);
  }
  if (dual_route) {
    const utils::DispatchRange prefill_range =
        record_k16 ? k16_range : materialized_range;
    route_group = graph.register_dispatch_route_group(
        {prefill_range, fd_resources.dispatch_range});
  }

  auto refresh_state = [compute_live_state,
                        q_id,
                        out_id,
                        dual_route,
                        record_k16,
                        record_materialized,
                        record_fd,
                        fixed_use_fd = initial_state.use_fd,
                        route_group,
                        uc_k_idx,
                        uc_v_idx,
                        qk_idx,
                        softmax_idx,
                        av_idx,
                        k16_idx,
                        uc_k_buf,
                        uc_v_buf,
                        qk_buf,
                        softmax_buf,
                        av_buf,
                        k16_buf,
                        fd_resources](WebGPUGraph& gr) {
    const SdpaLiveState state = compute_live_state(gr);

    wgpuQueueWriteBuffer(
        gr.queue(),
        uc_k_buf,
        0,
        &state.update_cache,
        sizeof(state.update_cache));
    wgpuQueueWriteBuffer(
        gr.queue(),
        uc_v_buf,
        0,
        &state.update_cache,
        sizeof(state.update_cache));
    if (record_materialized) {
      wgpuQueueWriteBuffer(
          gr.queue(),
          qk_buf,
          0,
          &state.attn_weights,
          sizeof(state.attn_weights));
      wgpuQueueWriteBuffer(
          gr.queue(), softmax_buf, 0, &state.softmax, sizeof(state.softmax));
      wgpuQueueWriteBuffer(
          gr.queue(), av_buf, 0, &state.compute_out, sizeof(state.compute_out));
    }
    if (record_k16) {
      wgpuQueueWriteBuffer(
          gr.queue(),
          k16_buf,
          0,
          &state.streaming_k16,
          sizeof(state.streaming_k16));
    }
    if (record_fd) {
      write_sdpa_fd_decode_uniforms(gr.queue(), fd_resources, state.fd);
    }

    gr.dispatch_at(uc_k_idx).workgroup_count_x = state.update_cache_grid.x;
    gr.dispatch_at(uc_k_idx).workgroup_count_y = state.update_cache_grid.y;
    gr.dispatch_at(uc_v_idx).workgroup_count_x = state.update_cache_grid.x;
    gr.dispatch_at(uc_v_idx).workgroup_count_y = state.update_cache_grid.y;
    if (dual_route) {
      const size_t active_route = state.use_fd ? 1 : 0;
      const std::vector<utils::WgCount> active_grids = state.use_fd
          ? std::vector<
                utils::WgCount>{state.fd.split_grid, state.fd.reduce_grid}
          : (record_k16
                 ? std::vector<utils::WgCount>{state.streaming_k16_grid}
                 : std::vector<utils::WgCount>{
                       state.qk_grid, state.softmax_grid, state.av_grid});
      gr.select_dispatch_route(route_group, active_route, active_grids);
    } else if (state.use_fd) {
      if (!fixed_use_fd) {
        throw std::runtime_error("WebGPU sdpa: static route changed");
      }
      gr.dispatch_at(fd_resources.dispatch_range.begin).workgroup_count_x =
          state.fd.split_grid.x;
      gr.dispatch_at(fd_resources.dispatch_range.begin).workgroup_count_y =
          state.fd.split_grid.y;
      gr.dispatch_at(fd_resources.dispatch_range.begin + 1).workgroup_count_x =
          state.fd.reduce_grid.x;
      gr.dispatch_at(fd_resources.dispatch_range.begin + 1).workgroup_count_y =
          state.fd.reduce_grid.y;
    } else if (record_k16) {
      if (fixed_use_fd) {
        throw std::runtime_error("WebGPU sdpa: static route changed");
      }
      gr.dispatch_at(k16_idx).workgroup_count_x = state.streaming_k16_grid.x;
      gr.dispatch_at(k16_idx).workgroup_count_y = state.streaming_k16_grid.y;
    } else {
      if (fixed_use_fd) {
        throw std::runtime_error("WebGPU sdpa: static route changed");
      }
      gr.dispatch_at(qk_idx).workgroup_count_x = state.qk_grid.x;
      gr.dispatch_at(qk_idx).workgroup_count_y = state.qk_grid.y;
      gr.dispatch_at(softmax_idx).workgroup_count_x = state.softmax_grid.x;
      gr.dispatch_at(softmax_idx).workgroup_count_y = state.softmax_grid.y;
      gr.dispatch_at(av_idx).workgroup_count_x = state.av_grid.x;
      gr.dispatch_at(av_idx).workgroup_count_y = state.av_grid.y;
    }
    gr.set_cur_dims(out_id, gr.cur_dims(q_id));
  };

  refresh_state(graph);
  graph.add_tensor_resize_hook(q_id, refresh_state);
  if (dynamic_pos) {
    graph.add_resize_hook(input_pos_id, refresh_state);
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(sdpa_with_kv_cache.default, sdpa_with_kv_cache_impl);
}

} // namespace executorch::backends::webgpu

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>
#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_coop4_bicol_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_shmem_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_steel_half_pwdq_f16acc_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_steel_half_pwdq_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_steel_half_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_steel_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_steel_bk64_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned, 32 bytes).
struct Q4gswParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(Q4gswParams) == 32, "Q4gswParams must be 32 bytes");

// Register-tile dims; MUST match TM/TN in q4gsw_linear.wgsl.
constexpr int64_t kQ4gswTileM = 4;
constexpr int64_t kQ4gswTileN = 4;

// Shmem-GEMM tile dims; MUST match WG_M/WG_N in q4gsw_linear_gemm_shmem.wgsl.
constexpr int64_t kQ4gswShmemTileM = 32;
constexpr int64_t kQ4gswShmemTileN = 32;
// Prefill route: shmem GEMM wins large K/N; the square 2048^2 (q/o proj,
// N=2048) also wins on shmem (+~50% Canary, 10-warm/50-run; the earlier -27%
// did not reproduce), so route it via a lower N threshold while k/v (N=512)
// stays on register-tiled.
constexpr uint32_t kQ4gswShmemMinDim = 4096u;
constexpr uint32_t kQ4gswShmemNMinDim = 2048u;

// steel GEMM: 64x64 tile, 256 threads (16x16), fixed wg (no override).
constexpr uint32_t kQ4gswSteelTile = 64u;
constexpr uint32_t kQ4gswSteelBK = 16u;
constexpr uint32_t kQ4gswSteelBK64 = 64u;
constexpr uint32_t kQ4gswSteelInvocations = 256u;

// One workgroup per (tile_m x tile_n) tile, no grid-stride: throw when the tile
// count would exceed the 1D dispatch limit. Shared by the steel + shmem GEMM
// routes; `kind` names the route in the error message.
uint32_t tiled_wg_count(
    WGPUDevice device,
    uint32_t m,
    uint32_t n,
    int64_t tile_m,
    int64_t tile_n,
    const char* op_name,
    const char* kind) {
  const int64_t total_wgs =
      utils::div_up<int64_t>(m, tile_m) * utils::div_up<int64_t>(n, tile_n);
  if (total_wgs > static_cast<int64_t>(utils::queried_max_workgroups(device))) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name + ": " + kind +
        " tile count exceeds the 1D dispatch limit");
  }
  return static_cast<uint32_t>(total_wgs);
}

// steel needs 256-thread workgroups; fail-closed (query ok AND >=256).
bool steel_supported(WGPUDevice device) {
  WGPULimits limits = {};
  return wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
      limits.maxComputeInvocationsPerWorkgroup >= kQ4gswSteelInvocations;
}

bool steel_bk64_eligible(
    WGPUDevice device,
    uint32_t K,
    uint32_t N,
    uint32_t group_size,
    bool has_bias) {
  WGPULimits limits = {};
  if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) {
    return false;
  }
  const WebGPUContext* context = get_default_webgpu_context();
  return utils::is_q4gsw_bk64_eligible(
      K,
      N,
      group_size,
      has_bias,
      context != nullptr && context->shader_f16_supported,
      limits.maxComputeInvocationsPerWorkgroup,
      limits.maxComputeWorkgroupStorageSize);
}

// Not grid-strided: 0 (fall back) when K%BK != 0 or over the 1D dispatch limit.
uint32_t
steel_workgroup_count(WGPUDevice device, uint32_t m, uint32_t n, uint32_t K) {
  if (K % kQ4gswSteelBK != 0u) {
    return 0u;
  }
  const uint64_t total =
      static_cast<uint64_t>((m + kQ4gswSteelTile - 1u) / kQ4gswSteelTile) *
      static_cast<uint64_t>((n + kQ4gswSteelTile - 1u) / kQ4gswSteelTile);
  const uint32_t max_count = utils::queried_max_workgroups(device);
  return (total == 0u || total > max_count) ? 0u : static_cast<uint32_t>(total);
}

uint32_t steel_bk64_workgroup_count(
    WGPUDevice device,
    uint32_t m,
    uint32_t n,
    uint32_t K) {
  if (K % kQ4gswSteelBK64 != 0u) {
    return 0u;
  }
  return steel_workgroup_count(device, m, n, K);
}

// Workgroup count for a linear_q4gsw dispatch (bicol GEMV / shmem GEMM / tiled
// GEMM), with the range/limit guards shared by the build-time path and the
// resize hook. use_gemv/use_shmem_gemm are the build-time routing decision (the
// shader/pipeline is fixed at build); the resize hook re-runs this with live m.
uint32_t compute_q4gsw_workgroup_count(
    WGPUDevice device,
    bool use_gemv,
    bool use_bk64,
    bool use_steel,
    bool use_shmem_gemm,
    uint32_t m,
    uint32_t n,
    uint32_t K,
    uint32_t wg_size,
    const char* op_name) {
  if (use_gemv) {
    // bicol: fixed 64 lanes, 2 output columns/workgroup, grid-strided over
    // ceil(N/2) column-pairs (M == 1 on this decode path).
    const uint64_t pairs = (static_cast<uint64_t>(n) + 1u) / 2u;
    if (pairs == 0u || pairs > UINT32_MAX) {
      throw std::runtime_error(
          std::string("WebGPU ") + op_name + ": N/2 out of range");
    }
    const uint32_t wgc =
        utils::clamp_workgroup_count(device, static_cast<uint32_t>(pairs));
    if (wgc == 0u) {
      throw std::runtime_error(
          std::string("WebGPU ") + op_name + ": zero GEMV dispatch");
    }
    return wgc;
  }
  if (use_bk64) {
    const uint32_t count = steel_bk64_workgroup_count(device, m, n, K);
    if (count == 0u) {
      throw std::runtime_error(
          std::string("WebGPU ") + op_name + ": invalid BK64 dispatch");
    }
    return count;
  }
  if (use_steel) {
    // steel: one workgroup per 64x64 tile. Over-limit THROWS here -- unlike the
    // build-time steel_workgroup_count, which returns 0 so the caller falls
    // back to shmem/tiled. The routed kernel is baked into the pipeline at
    // build, so the resize path cannot switch kernels for a larger live M.
    return tiled_wg_count(
        device, m, n, kQ4gswSteelTile, kQ4gswSteelTile, op_name, "steel GEMM");
  }
  if (use_shmem_gemm) {
    // shmem GEMM: one workgroup per tile.
    return tiled_wg_count(
        device,
        m,
        n,
        kQ4gswShmemTileM,
        kQ4gswShmemTileN,
        op_name,
        "shmem GEMM");
  }
  const int64_t total_tiles = utils::div_up<int64_t>(m, kQ4gswTileM) *
      utils::div_up<int64_t>(n, kQ4gswTileN);
  if (total_tiles > static_cast<int64_t>(UINT32_MAX)) {
    throw std::runtime_error(
        std::string("WebGPU ") + op_name +
        ": tile count exceeds the 1D dispatch limit");
  }
  return utils::compute_1d_workgroup_count(
      device, static_cast<uint32_t>(total_tiles), wg_size, op_name);
}

struct Q4gswExecutionState {
  Q4gswParams params;
  std::vector<int64_t> output_dims;
  size_t active_route;
  utils::WgCount active_grid;
};

constexpr size_t kQ4gswBicolRoute = 0;
constexpr size_t kQ4gswBk64Route = 1;
constexpr size_t kQ4gswPrefillRoute = 2;

Q4gswExecutionState make_q4gsw_execution_state(
    WGPUDevice device,
    const std::vector<int64_t>& input_dims,
    uint32_t max_m,
    uint32_t K,
    uint32_t N,
    uint32_t K_packed,
    uint32_t gs,
    uint32_t padded_N,
    uint32_t has_bias,
    uint32_t wg_size,
    bool use_single_gemv,
    bool use_dual_route,
    bool record_bk64_route,
    bool bk64_eligible,
    bool prefill_use_steel,
    bool prefill_use_shmem_gemm) {
  if (input_dims.empty()) {
    throw std::runtime_error("WebGPU linear_q4gsw(resize): empty input dims");
  }
  const uint64_t numel = utils::numel_of(input_dims);
  if (numel % static_cast<uint64_t>(K) != 0u) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw(resize): live input numel not a multiple of K");
  }
  const uint64_t live_m = numel / static_cast<uint64_t>(K);
  if (live_m == 0u) {
    throw std::runtime_error("WebGPU linear_q4gsw(resize): live M == 0");
  }
  if (live_m > max_m) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw(resize): live M exceeds the build-time max");
  }
  const uint32_t m = static_cast<uint32_t>(live_m);
  const bool use_gemv = use_single_gemv || (use_dual_route && m == 1u);
  const bool use_bk64 = !use_gemv && bk64_eligible &&
      utils::is_q4gsw_bk64_live_m(m) &&
      steel_bk64_workgroup_count(device, m, N, K) > 0u;
  const uint32_t workgroup_count = compute_q4gsw_workgroup_count(
      device,
      use_gemv,
      use_bk64,
      !use_gemv && !use_bk64 && prefill_use_steel,
      !use_gemv && !use_bk64 && prefill_use_shmem_gemm,
      m,
      N,
      K,
      wg_size,
      "linear_q4gsw(resize)");

  Q4gswExecutionState state = {};
  state.params.M = m;
  state.params.N = N;
  state.params.K = K;
  state.params.K_packed = K_packed;
  state.params.group_size = gs;
  state.params.padded_N = padded_N;
  state.params.has_bias = has_bias;
  state.output_dims = input_dims;
  state.output_dims.back() = static_cast<int64_t>(N);
  state.active_route = use_dual_route
      ? (use_gemv ? kQ4gswBicolRoute
                  : (record_bk64_route
                         ? (use_bk64 ? kQ4gswBk64Route : kQ4gswPrefillRoute)
                         : 1u))
      : 0u;
  state.active_grid = {workgroup_count, 1u};
  return state;
}

// et_vk.linear_q4gsw args: [in, weight, scales, group_size, bias, out].
void q4gsw_linear_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int scales_id = args.at(2);
  const int group_size_id = args.at(3);
  const int bias_id = args.at(4);
  const int out_id = args.at(5);

  WGPUDevice device = graph.device();

  const auto& in = graph.get_tensor(in_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.empty() || weight.dims.size() < 2 || scales.dims.size() < 2) {
    throw std::runtime_error("WebGPU linear_q4gsw: malformed input dims");
  }

  // Shapes from the tensors' own dims (no dtype field at runtime).
  const uint32_t K = static_cast<uint32_t>(in.dims.back());
  if (K == 0) {
    throw std::runtime_error("WebGPU linear_q4gsw: K == 0");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint32_t M = static_cast<uint32_t>(in_numel / K);
  if (in_numel % K != 0) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw: input numel not a multiple of K");
  }
  const uint32_t N = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t K_packed = static_cast<uint32_t>(weight.dims[1]);
  const uint32_t num_groups = static_cast<uint32_t>(scales.dims[0]);
  const uint32_t padded_N = static_cast<uint32_t>(scales.dims[1]);
  if (M == 0 || N == 0) {
    throw std::runtime_error("WebGPU linear_q4gsw: M or N == 0");
  }
  // int4 packing is 2 nibbles/byte, so K_packed must be ceil(K/2) (guards OOB).
  if (K_packed != (K + 1) / 2) {
    throw std::runtime_error("WebGPU linear_q4gsw: K_packed must be ceil(K/2)");
  }
  // Weight is read as array<u32>; a non-multiple-of-4 byte count over-reads.
  if ((static_cast<uint64_t>(N) * K_packed) % 4u != 0u) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw: N*K_packed must be a multiple of 4 (u32-packed)");
  }

  // fp32-only byte-size guards (no runtime dtype); fp16 scales -> bail.
  const uint64_t scales_numel =
      static_cast<uint64_t>(num_groups) * static_cast<uint64_t>(padded_N);
  const uint64_t weight_numel =
      static_cast<uint64_t>(N) * static_cast<uint64_t>(K_packed);
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != static_cast<uint64_t>(M) * N * sizeof(float) ||
      scales.nbytes != scales_numel * sizeof(float) ||
      weight.nbytes != weight_numel) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw: fp32-only (byte-size mismatch)");
  }

  int64_t group_size = 0;
  if (graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int) {
    group_size = graph.get_int(group_size_id);
  }
  if (group_size <= 0) {
    throw std::runtime_error("WebGPU linear_q4gsw: group_size <= 0");
  }
  // scales is indexed [(k/group_size)*padded_N + n]; guard the table bounds.
  const uint32_t gs = static_cast<uint32_t>(group_size);
  if (num_groups < (K + gs - 1u) / gs || padded_N < N) {
    throw std::runtime_error(
        "WebGPU linear_q4gsw: scales dims too small for K/N");
  }

  // Optional bias: real buffer if present, else a dummy for the fixed layout.
  uint32_t has_bias = 0;
  WGPUBuffer bias_buffer = nullptr;
  uint64_t bias_size = 4;
  if (graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor) {
    const auto& bias = graph.get_tensor(bias_id);
    if (bias.buffer == nullptr || bias.nbytes < N * sizeof(float)) {
      throw std::runtime_error(
          "WebGPU linear_q4gsw: bias present but null/undersized");
    }
    has_bias = 1;
    bias_buffer = bias.buffer;
    bias_size = bias.nbytes;
  }
  if (bias_buffer == nullptr) {
    bias_buffer = graph.create_scratch_buffer(4);
  }

  // M==1 -> bicol GEMV; M>1 -> BK64 for exact Llama projections/M values,
  // otherwise steel GEMM (preferred), shmem, or tiled.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ4gswLinearWorkgroupSizeX);
  const bool bicol_eligible = K % 8u == 0u && gs % 8u == 0u;
  const bool use_gemv = M == 1u && bicol_eligible;
  const bool use_dual_route = utils::should_record_q4gsw_dual_route(
      M, bicol_eligible, graph.has_dynamic_shapes());
  const bool bk64_eligible =
      steel_bk64_eligible(device, K, N, gs, has_bias != 0u);
  const bool record_bk64_route = use_dual_route && bk64_eligible && M >= 128u;
  const bool use_bk64 = !use_gemv && bk64_eligible &&
      utils::is_q4gsw_bk64_live_m(M) &&
      steel_bk64_workgroup_count(device, M, N, K) > 0u;
  // GEMV (bicol) is a pow2 tree reduction; compute its size only when used.
  const uint32_t gemv_wg_size = (use_gemv || use_dual_route)
      ? utils::clamp_workgroup_size_pow2(
            device, kQ4gswLinearCoop4BicolWorkgroupSizeX)
      : 0u;
  // steel (256-thread) is the preferred M>1 prefill GEMM; 0 count = ineligible.
  const bool use_steel = !use_gemv && steel_supported(device) &&
      steel_workgroup_count(device, M, N, K) > 0u;
  // shmem GEMM is now a FALLBACK, not dead: steel shadows it whenever eligible,
  // so shmem only wins when steel is ineligible (K % 16 != 0, or a
  // <256-invocation device such as SwiftShader) and the shape still hits the
  // large K/N thresholds; otherwise the register-tiled path handles it.
  const bool use_shmem_gemm = !use_gemv && !use_steel &&
      (K >= kQ4gswShmemMinDim || N >= kQ4gswShmemNMinDim);
  const char* prefill_shader_src = use_steel ? kQ4gswLinearGemmSteelWGSL
      : use_shmem_gemm                       ? kQ4gswLinearGemmShmemWGSL
                                             : kQ4gswLinearWGSL;
  const char* shader_src = use_gemv ? kQ4gswLinearCoop4BicolWGSL
      : use_bk64                    ? kQ4gswSteelBk64WGSL
      : use_steel                   ? kQ4gswLinearGemmSteelWGSL
      : use_shmem_gemm              ? kQ4gswLinearGemmShmemWGSL
                                    : kQ4gswLinearWGSL;
  // f16-multiply steel: only when the device negotiated shader-f16; else the
  // f32 steel kernel runs (fail-closed). Same bindings and tile.
  if (use_steel) {
    const WebGPUContext* ctx = get_default_webgpu_context();
    if (ctx != nullptr && ctx->shader_f16_supported) {
      // Packed-word dequant: bit-exact to the steel `half` kernel but loads
      // each u32 weight word once + hoists the per-column scale (half re-reads
      // them ~8x/~16x). Needs group_size % BK == 0 so the hoisted scale is
      // constant across the BK tile; else the per-nibble `half` kernel.
      prefill_shader_src = (gs % kQ4gswSteelBK == 0u)
          ? kQ4gswLinearGemmSteelHalfPwdqWGSL
          : kQ4gswLinearGemmSteelHalfWGSL;
      if (!use_bk64) {
        shader_src = prefill_shader_src;
      }
    }
  }
  // f16-accumulate: pwdq staging with an f16 register accumulator.
  // Lossy (f16 accumulate over K) -> opt-in via the enable_f16_accumulate_gemm
  // runtime spec (default off), gated on the negotiated shader-f16 feature and
  // group_size % BK == 0 (same hoisted-scale requirement as pwdq). Overrides
  // the f32-accumulate steel kernels.
  if (use_steel && graph.f16_accumulate_gemm() && (gs % kQ4gswSteelBK == 0u)) {
    const WebGPUContext* ctx = get_default_webgpu_context();
    if (ctx != nullptr && ctx->shader_f16_supported) {
      prefill_shader_src = kQ4gswLinearGemmSteelHalfPwdqF16accWGSL;
      if (!use_bk64) {
        shader_src = prefill_shader_src;
      }
    }
  }

  const Q4gswExecutionState initial_state = make_q4gsw_execution_state(
      device,
      in.dims,
      M,
      K,
      N,
      K_packed,
      gs,
      padded_N,
      has_bias,
      wg_size,
      use_gemv,
      use_dual_route,
      record_bk64_route,
      bk64_eligible,
      use_steel,
      use_shmem_gemm);

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(Q4gswParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(Q4gswParams));
  std::memcpy(mapped, &initial_state.params, sizeof(Q4gswParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(Q4gswParams));

  // Bind group layout: out (rw) + in/weight/scales/bias (ro storage) + uniform.
  WGPUBindGroupLayoutEntry entries[6] = {};
  entries[0].binding = 0;
  entries[0].visibility = WGPUShaderStage_Compute;
  entries[0].buffer.type = WGPUBufferBindingType_Storage;
  for (uint32_t i = 1; i <= 4; i++) {
    entries[i].binding = i;
    entries[i].visibility = WGPUShaderStage_Compute;
    entries[i].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  }
  entries[5].binding = 5;
  entries[5].visibility = WGPUShaderStage_Compute;
  entries[5].buffer.type = WGPUBufferBindingType_Uniform;

  WGPUBindGroupLayoutDescriptor bgl_desc = {};
  bgl_desc.entryCount = 6;
  bgl_desc.entries = entries;
  WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);

  WGPUPipelineLayoutDescriptor pl_desc = {};
  pl_desc.bindGroupLayoutCount = 1;
  pl_desc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(device, &pl_desc);

  WGPUBindGroupEntry bg_entries[6] = {};
  bg_entries[0].binding = 0;
  bg_entries[0].buffer = out.buffer;
  bg_entries[0].size = out.nbytes;
  bg_entries[1].binding = 1;
  bg_entries[1].buffer = in.buffer;
  bg_entries[1].size = in.nbytes;
  bg_entries[2].binding = 2;
  bg_entries[2].buffer = weight.buffer;
  bg_entries[2].size = weight.nbytes;
  bg_entries[3].binding = 3;
  bg_entries[3].buffer = scales.buffer;
  bg_entries[3].size = scales.nbytes;
  bg_entries[4].binding = 4;
  bg_entries[4].buffer = bias_buffer;
  bg_entries[4].size = bias_size;
  bg_entries[5].binding = 5;
  bg_entries[5].buffer = uniform_buffer;
  bg_entries[5].size = sizeof(Q4gswParams);

  WGPUBindGroupDescriptor bg_desc = {};
  bg_desc.layout = bgl;
  bg_desc.entryCount = 6;
  bg_desc.entries = bg_entries;
  WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);

  auto make_pipeline = [&](const char* source,
                           bool fixed_wg,
                           uint32_t override_wg_size) {
    WGPUShaderSourceWGSL wgsl_desc = {};
    wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_desc.code = {source, WGPU_STRLEN};
    WGPUShaderModuleDescriptor shader_desc = {};
    shader_desc.nextInChain = &wgsl_desc.chain;
    WGPUShaderModule shader =
        wgpuDeviceCreateShaderModule(device, &shader_desc);

    WGPUConstantEntry wg_size_constant = {};
    wg_size_constant.key = {"wg_size", WGPU_STRLEN};
    wg_size_constant.value = static_cast<double>(override_wg_size);
    WGPUComputePipelineDescriptor pipeline_desc = {};
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
    pipeline_desc.compute.constantCount = fixed_wg ? 0u : 1u;
    pipeline_desc.compute.constants = fixed_wg ? nullptr : &wg_size_constant;
    WGPUComputePipeline pipeline =
        wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    wgpuShaderModuleRelease(shader);
    return pipeline;
  };

  const bool fixed_prefill_wg = use_steel || use_shmem_gemm;
  const char* prefill_label = use_steel ? "linear_q4gsw_steel"
      : use_shmem_gemm                  ? "linear_q4gsw_shmem"
                                        : "linear_q4gsw_tiled";
  size_t dispatch_idx = 0;
  size_t route_group = 0;
  if (use_dual_route) {
    // Each recorded dispatch owns one bind-group reference.
    wgpuBindGroupAddRef(bind_group);
    if (record_bk64_route) {
      wgpuBindGroupAddRef(bind_group);
    }
    WGPUComputePipeline bicol_pipeline =
        make_pipeline(kQ4gswLinearCoop4BicolWGSL, false, gemv_wg_size);
    const size_t bicol_idx = graph.add_dispatch(
        {bicol_pipeline,
         bind_group,
         initial_state.active_grid.x,
         "linear_q4gsw_coop4_bicol"});
    size_t bk64_idx = 0;
    if (record_bk64_route) {
      WGPUComputePipeline bk64_pipeline =
          make_pipeline(kQ4gswSteelBk64WGSL, true, 0u);
      bk64_idx = graph.add_dispatch(
          {bk64_pipeline,
           bind_group,
           initial_state.active_grid.x,
           "linear_q4gsw_bk64"});
    }
    WGPUComputePipeline prefill_pipeline =
        make_pipeline(prefill_shader_src, fixed_prefill_wg, wg_size);
    const size_t prefill_idx = graph.add_dispatch(
        {prefill_pipeline,
         bind_group,
         initial_state.active_grid.x,
         prefill_label});
    if (record_bk64_route) {
      route_group = graph.register_dispatch_route_group(
          {{bicol_idx, bicol_idx + 1},
           {bk64_idx, bk64_idx + 1},
           {prefill_idx, prefill_idx + 1}});
    } else {
      route_group = graph.register_dispatch_route_group(
          {{bicol_idx, bicol_idx + 1}, {prefill_idx, prefill_idx + 1}});
    }
    graph.select_dispatch_route(
        route_group, initial_state.active_route, {initial_state.active_grid});
  } else {
    const bool fixed_wg = use_gemv ? false : (use_bk64 || fixed_prefill_wg);
    WGPUComputePipeline pipeline =
        make_pipeline(shader_src, fixed_wg, use_gemv ? gemv_wg_size : wg_size);
    dispatch_idx = graph.add_dispatch(
        {pipeline,
         bind_group,
         initial_state.active_grid.x,
         use_gemv ? "linear_q4gsw_coop4_bicol"
                  : (use_bk64 ? "linear_q4gsw_bk64" : prefill_label)});
  }

  // Dynamic shapes: recompute one shared Params block and select exactly one
  // writer. The prefill pipeline remains the route chosen from max M.
  graph.add_tensor_resize_hook(
      in_id,
      [in_id,
       out_id,
       M,
       K,
       N,
       K_packed,
       gs,
       padded_N,
       has_bias,
       wg_size,
       use_gemv,
       use_dual_route,
       record_bk64_route,
       bk64_eligible,
       use_steel,
       use_shmem_gemm,
       dispatch_idx,
       route_group,
       uniform_buffer](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const Q4gswExecutionState state = make_q4gsw_execution_state(
            g.device(),
            d,
            M,
            K,
            N,
            K_packed,
            gs,
            padded_N,
            has_bias,
            wg_size,
            use_gemv,
            use_dual_route,
            record_bk64_route,
            bk64_eligible,
            use_steel,
            use_shmem_gemm);
        wgpuQueueWriteBuffer(
            g.queue(), uniform_buffer, 0, &state.params, sizeof(state.params));
        if (use_dual_route) {
          g.select_dispatch_route(
              route_group, state.active_route, {state.active_grid});
        } else {
          auto& dispatch = g.dispatch_at(dispatch_idx);
          dispatch.workgroup_count_x = state.active_grid.x;
          dispatch.workgroup_count_y = state.active_grid.y;
        }
        g.set_cur_dims(out_id, state.output_dims);
      });

  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_q4gsw.default, q4gsw_linear_impl);
}

} // namespace executorch::backends::webgpu

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
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_coop4_bicol_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_gemm_shmem_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/quantized_linear/q4gsw_linear_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

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

  // M==1 -> bicol GEMV; M>1 -> shmem GEMM (large K/N) else tiled GEMM.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ4gswLinearWorkgroupSizeX);
  const bool use_gemv = (M == 1u && K % 8u == 0u && gs % 8u == 0u);
  const bool use_shmem_gemm =
      !use_gemv && (K >= kQ4gswShmemMinDim || N >= kQ4gswShmemNMinDim);
  const char* shader_src = use_gemv ? kQ4gswLinearCoop4BicolWGSL
      : use_shmem_gemm              ? kQ4gswLinearGemmShmemWGSL
                                    : kQ4gswLinearWGSL;
  uint32_t workgroup_count;
  if (use_gemv) {
    // bicol: fixed 64 lanes, 2 output columns/workgroup, grid-strided over
    // ceil(N/2) column-pairs (M == 1 on this decode path).
    const uint64_t pairs = (static_cast<uint64_t>(N) + 1u) / 2u;
    if (pairs == 0u || pairs > UINT32_MAX) {
      throw std::runtime_error("WebGPU linear_q4gsw: N/2 out of range");
    }
    workgroup_count =
        utils::clamp_workgroup_count(device, static_cast<uint32_t>(pairs));
    if (workgroup_count == 0u) {
      throw std::runtime_error("WebGPU linear_q4gsw: zero GEMV dispatch");
    }
  } else if (use_shmem_gemm) {
    // shmem GEMM: one workgroup per tile, no grid-stride -> throw over limit.
    const int64_t total_wgs = utils::div_up<int64_t>(M, kQ4gswShmemTileM) *
        utils::div_up<int64_t>(N, kQ4gswShmemTileN);
    WGPULimits limits = {};
    const uint32_t max_wgs =
        wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success &&
            limits.maxComputeWorkgroupsPerDimension > 0
        ? limits.maxComputeWorkgroupsPerDimension
        : 65535u;
    if (total_wgs > static_cast<int64_t>(max_wgs)) {
      throw std::runtime_error(
          "WebGPU linear_q4gsw: shmem GEMM tile count exceeds the 1D dispatch limit");
    }
    workgroup_count = static_cast<uint32_t>(total_wgs);
  } else {
    const int64_t total_tiles = utils::div_up<int64_t>(M, kQ4gswTileM) *
        utils::div_up<int64_t>(N, kQ4gswTileN);
    if (total_tiles > static_cast<int64_t>(UINT32_MAX)) {
      throw std::runtime_error(
          "WebGPU linear_q4gsw: tile count exceeds the 1D dispatch limit");
    }
    workgroup_count = utils::compute_1d_workgroup_count(
        device, static_cast<uint32_t>(total_tiles), wg_size, "linear_q4gsw");
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

  Q4gswParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;
  params.group_size = gs;
  params.padded_N = padded_N;
  params.has_bias = has_bias;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(Q4gswParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(Q4gswParams));
  std::memcpy(mapped, &params, sizeof(Q4gswParams));
  wgpuBufferUnmap(uniform_buffer);
  graph.add_uniform_buffer_bytes(sizeof(Q4gswParams));

  WGPUShaderSourceWGSL wgsl_desc = {};
  wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
  wgsl_desc.code = {shader_src, WGPU_STRLEN};
  WGPUShaderModuleDescriptor shader_desc = {};
  shader_desc.nextInChain = &wgsl_desc.chain;
  WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &shader_desc);

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

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUComputePipelineDescriptor pipeline_desc = {};
  pipeline_desc.layout = pipeline_layout;
  pipeline_desc.compute.module = shader;
  pipeline_desc.compute.entryPoint = {"main", WGPU_STRLEN};
  // Only the tiled GEMM has a wg_size override; GEMV + shmem are fixed 64.
  const bool fixed_wg = use_gemv || use_shmem_gemm;
  pipeline_desc.compute.constantCount = fixed_wg ? 0u : 1u;
  pipeline_desc.compute.constants = fixed_wg ? nullptr : &wg_size_constant;
  WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

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

  graph.add_dispatch({pipeline, bind_group, workgroup_count, "linear_q4gsw"});

  wgpuShaderModuleRelease(shader);
  wgpuBindGroupLayoutRelease(bgl);
  wgpuPipelineLayoutRelease(pipeline_layout);
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_q4gsw.default, q4gsw_linear_impl);
}

} // namespace executorch::backends::webgpu

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
#include <executorch/backends/webgpu/runtime/ops/linear_dq8ca_q4gsw/linear_dq8ca_q4gsw_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Dq8caParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t K_packed;
  uint32_t group_size;
  uint32_t padded_N;
  uint32_t has_bias;
  uint32_t _pad;
};
static_assert(sizeof(Dq8caParams) == 32, "Dq8caParams must be 32 bytes");

constexpr int64_t kTileM = 4; // MUST match TM in linear_dq8ca_q4gsw.wgsl
constexpr int64_t kTileN = 4; // MUST match TN

// et_vk.linear_dq8ca_q4gsw args (mirrors Vulkan QuantizedLinear.cpp:760):
// [in, input_scale, input_zp, weight, weight_sums, weight_scales, group_size,
//  bias, out]. Dynamic per-row int8 activation quant x 4-bit-group symmetric
// weight. weight_sums (arg 4) is a perf shortcut; this v1 recomputes the sum
// inline so it is intentionally unused. Static-shape only (no resize hook yet).
void linear_dq8ca_q4gsw_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int input_scale_id = args.at(1);
  const int input_zp_id = args.at(2);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int group_size_id = args.at(6);
  const int bias_id = args.at(7);
  const int out_id = args.at(8);

  WGPUDevice device = graph.device();
  const auto& in = graph.get_tensor(in_id);
  const auto& input_scale = graph.get_tensor(input_scale_id);
  const auto& input_zp = graph.get_tensor(input_zp_id);
  const auto& weight = graph.get_tensor(weight_id);
  const auto& scales = graph.get_tensor(scales_id);
  const auto& out = graph.get_tensor(out_id);

  if (in.dims.empty() || weight.dims.size() < 2 || scales.dims.size() < 2) {
    throw std::runtime_error("linear_dq8ca_q4gsw: malformed dims");
  }
  if (in.buffer == nullptr || input_scale.buffer == nullptr ||
      input_zp.buffer == nullptr || weight.buffer == nullptr ||
      scales.buffer == nullptr || out.buffer == nullptr) {
    throw std::runtime_error("linear_dq8ca_q4gsw: null buffer binding");
  }

  const uint32_t K = static_cast<uint32_t>(in.dims.back());
  if (K == 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: K == 0");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  if (in_numel % K != 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: input numel % K != 0");
  }
  const uint32_t M = static_cast<uint32_t>(in_numel / K);
  const uint32_t N = static_cast<uint32_t>(weight.dims[0]);
  const uint32_t K_packed = static_cast<uint32_t>(weight.dims[1]);
  const uint32_t num_groups = static_cast<uint32_t>(scales.dims[0]);
  const uint32_t padded_N = static_cast<uint32_t>(scales.dims[1]);
  if (M == 0 || N == 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: M or N == 0");
  }
  if (K_packed != (K + 1) / 2) {
    throw std::runtime_error("linear_dq8ca_q4gsw: K_packed must be ceil(K/2)");
  }
  if ((static_cast<uint64_t>(N) * K_packed) % 4u != 0u) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: N*K_packed must be a multiple of 4 (u32-packed)");
  }

  int64_t group_size = 0;
  if (graph.get_value_type(group_size_id) == WebGPUGraph::ValueType::Int) {
    group_size = graph.get_int(group_size_id);
  }
  if (group_size <= 0) {
    throw std::runtime_error("linear_dq8ca_q4gsw: group_size <= 0");
  }
  const uint32_t gs = static_cast<uint32_t>(group_size);

  // fp32-only byte guards; per-row scale (f32[M]) + zp (int8[M]).
  if (in.nbytes != in_numel * sizeof(float) ||
      out.nbytes != static_cast<uint64_t>(M) * N * sizeof(float) ||
      scales.nbytes !=
          static_cast<uint64_t>(num_groups) * padded_N * sizeof(float) ||
      weight.nbytes != static_cast<uint64_t>(N) * K_packed) {
    throw std::runtime_error("linear_dq8ca_q4gsw: fp32/byte-size mismatch");
  }
  // Per-row activation scale (fp32[M]) + zp (int8[M], packed 4/word in-shader).
  if (input_scale.nbytes != static_cast<uint64_t>(M) * sizeof(float) ||
      !input_zp.is_int8 || input_zp.nbytes != static_cast<uint64_t>(M)) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: input scale fp32[M] / zp int8[M] required");
  }
  // int8 zp is bound word-aligned over a max(nbytes,4) buffer; M in {5,6,7,...}
  // would bind past the buffer. Mirrors the choose_qparams_affine producer
  // guard.
  if (M > 4u && M % 4u != 0u) {
    throw std::runtime_error(
        "linear_dq8ca_q4gsw: num_rows must be <=4 or a multiple of 4");
  }
  if (num_groups < (K + gs - 1u) / gs || padded_N < N) {
    throw std::runtime_error("linear_dq8ca_q4gsw: scales dims too small");
  }

  uint32_t has_bias = 0;
  WGPUBuffer bias_buffer = nullptr;
  uint64_t bias_size = 4;
  if (graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor) {
    const auto& bias = graph.get_tensor(bias_id);
    if (bias.buffer != nullptr && bias.nbytes >= N * sizeof(float)) {
      has_bias = 1;
      bias_buffer = bias.buffer;
      bias_size = bias.nbytes;
    }
  }
  if (bias_buffer == nullptr) {
    bias_buffer = graph.create_scratch_buffer(4);
  }

  Dq8caParams params = {};
  params.M = M;
  params.N = N;
  params.K = K;
  params.K_packed = K_packed;
  params.group_size = gs;
  params.padded_N = padded_N;
  params.has_bias = has_bias;

  const int64_t total_tiles =
      utils::div_up<int64_t>(M, kTileM) * utils::div_up<int64_t>(N, kTileN);
  if (total_tiles > static_cast<int64_t>(UINT32_MAX)) {
    throw std::runtime_error("linear_dq8ca_q4gsw: tile count exceeds u32");
  }
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kLinearDq8caQ4gswWorkgroupSizeX);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device,
      static_cast<uint32_t>(total_tiles),
      wg_size,
      "linear_dq8ca_q4gsw");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Dq8caParams));
  graph.add_uniform_buffer_bytes(sizeof(Dq8caParams));

  // 0 out(rw), 1 in, 2 input_scale, 3 input_zp, 4 weight, 5 scales, 6 bias
  // (ro), 7 uniform.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kLinearDq8caQ4gswWGSL,
      {
          {0, WGPUBufferBindingType_Storage, out.buffer, out.nbytes},
          {1, WGPUBufferBindingType_ReadOnlyStorage, in.buffer, in.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           input_scale.buffer,
           input_scale.nbytes},
          // int8 zp bound as array<u32>; round to a multiple of 4 (buffer is
          // >=4 bytes).
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           input_zp.buffer,
           ((input_zp.nbytes + 3u) / 4u) * 4u},
          {4,
           WGPUBufferBindingType_ReadOnlyStorage,
           weight.buffer,
           weight.nbytes},
          {5,
           WGPUBufferBindingType_ReadOnlyStorage,
           scales.buffer,
           scales.nbytes},
          {6, WGPUBufferBindingType_ReadOnlyStorage, bias_buffer, bias_size},
          {7, WGPUBufferBindingType_Uniform, params_buf, sizeof(Dq8caParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "linear_dq8ca_q4gsw",
       workgroup_count.y});
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.linear_dq8ca_q4gsw.default, linear_dq8ca_q4gsw_impl);
}

} // namespace executorch::backends::webgpu

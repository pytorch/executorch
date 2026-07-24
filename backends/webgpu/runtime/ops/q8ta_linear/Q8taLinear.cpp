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
#include <executorch/backends/webgpu/runtime/ops/q8ta_linear/q8ta_linear_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taLinearParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  int32_t input_zero_point;
  int32_t output_zero_point;
  float input_scale;
  float inv_output_scale;
  uint32_t has_bias;
};
static_assert(
    sizeof(Q8taLinearParams) == 32,
    "Q8taLinearParams must match the WGSL Params struct (32 bytes)");

// int8-act x int8-weight linear->int8; mirrors Vulkan q8ta_linear (act=none).
void q8ta_linear_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [x,in_scale,in_zp,w,w_sums,w_scales,out_scale,out_zp,bias,act,out].
  const int in_id = args.at(0);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int bias_id = args.at(8);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(scales_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_linear: in/weight/scales/out not tensor");
  }
  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;
  // Only activation="none" is supported; relu-fused would compute wrong output.
  const int act_id = args.at(args.size() - 2);
  if (graph.get_value_type(act_id) != WebGPUGraph::ValueType::String ||
      graph.get_string(act_id) != "none") {
    throw std::runtime_error("q8ta_linear: only activation='none' supported");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& scales_tensor = graph.get_tensor(scales_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || weight_tensor.buffer == nullptr ||
      scales_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_linear: null buffer binding");
  }
  if (weight_tensor.dims.size() != 2) {
    throw std::runtime_error("q8ta_linear: weight must be 2D [N, K]");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));
  const double output_scale = graph.get_double(args.at(6));
  const int output_zero_point = graph.get_int(args.at(7));

  const uint64_t N = static_cast<uint64_t>(weight_tensor.dims.at(0));
  const uint64_t K = static_cast<uint64_t>(weight_tensor.dims.at(1));
  if (K == 0 || in_tensor.dims.empty() ||
      static_cast<uint64_t>(in_tensor.dims.back()) != K) {
    throw std::runtime_error("q8ta_linear: input last dim must equal K");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in_tensor.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  const uint64_t M = in_numel / K;
  if (M == 0 || N == 0 || out_tensor.dims.empty() ||
      static_cast<uint64_t>(out_tensor.dims.back()) != N) {
    throw std::runtime_error("q8ta_linear: bad M/N shape");
  }
  if (M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX) {
    throw std::runtime_error("q8ta_linear: dim exceeds u32");
  }
  // N % 4 == 0 makes each row's TN=4 outputs land in one aligned int8 word.
  if (N % 4 != 0) {
    throw std::runtime_error("q8ta_linear: N must be a multiple of 4");
  }
  // int8 x / weight / out bound as array<u32>: their numels must be %4 == 0.
  if (!in_tensor.is_int8 || in_tensor.nbytes != M * K || (M * K) % 4 != 0 ||
      !weight_tensor.is_int8 || weight_tensor.nbytes != N * K ||
      (N * K) % 4 != 0 || !out_tensor.is_int8 || out_tensor.nbytes != M * N) {
    throw std::runtime_error("q8ta_linear: int8 x/weight/out size mismatch");
  }
  if (scales_tensor.nbytes != N * sizeof(float)) {
    throw std::runtime_error("q8ta_linear: weight_scales must be fp32 [N]");
  }
  if (has_bias) {
    const auto& b = graph.get_tensor(bias_id);
    if (b.buffer == nullptr || b.nbytes != N * sizeof(float)) {
      throw std::runtime_error("q8ta_linear: bias must be fp32 [N]");
    }
  }

  Q8taLinearParams params = {};
  params.M = static_cast<uint32_t>(M);
  params.N = static_cast<uint32_t>(N);
  params.K = static_cast<uint32_t>(K);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);
  params.input_scale = static_cast<float>(input_scale);
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_output_scale = static_cast<float>(1.0 / output_scale);
  params.has_bias = has_bias ? 1u : 0u;

  const uint64_t n_tiles_u64 = ((M + 3) / 4) * (N / 4);
  if (n_tiles_u64 > UINT32_MAX) {
    throw std::runtime_error("q8ta_linear: dispatch tile count exceeds u32");
  }
  const uint32_t n_tiles = static_cast<uint32_t>(n_tiles_u64);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taLinearWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, n_tiles, wg_size, "q8ta_linear");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taLinearParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taLinearParams));

  // No-bias: bind scales as an unread placeholder (has_bias gates the read).
  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : scales_tensor.buffer;
  const uint64_t bias_size =
      has_bias ? graph.get_tensor(bias_id).nbytes : scales_tensor.nbytes;

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kQ8taLinearWGSL,
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
          {3,
           WGPUBufferBindingType_ReadOnlyStorage,
           scales_tensor.buffer,
           scales_tensor.nbytes},
          {4, WGPUBufferBindingType_ReadOnlyStorage, bias_buf, bias_size},
          {5,
           WGPUBufferBindingType_Uniform,
           params_buf,
           sizeof(Q8taLinearParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "q8ta_linear",
       workgroup_count.y});
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_linear.default, q8ta_linear_impl);
  // GEMV (M==1) variant: identical schema/semantics, handled by the same GEMM.
  WEBGPU_REGISTER_OP(et_vk.q8ta_linear_gemv.default, q8ta_linear_impl);
}

} // namespace executorch::backends::webgpu

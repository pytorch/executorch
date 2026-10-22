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
#include <executorch/backends/webgpu/runtime/ops/q8ta_conv2d/q8ta_conv2d_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taConvParams {
  uint32_t N;
  uint32_t IC;
  uint32_t H_in;
  uint32_t W_in;
  uint32_t OC;
  uint32_t H_out;
  uint32_t W_out;
  uint32_t Kh;
  uint32_t Kw;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t pad_h;
  uint32_t pad_w;
  uint32_t dil_h;
  uint32_t dil_w;
  uint32_t weight_row_stride;
  int32_t input_zero_point;
  int32_t output_zero_point;
  float input_scale;
  float inv_output_scale;
  uint32_t has_bias;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
static_assert(
    sizeof(Q8taConvParams) == 96,
    "Q8taConvParams must match the WGSL Params struct (96 bytes)");

std::pair<int64_t, int64_t> pair_or_throw(
    const std::vector<int64_t>& v,
    const char* msg) {
  if (v.size() != 2) {
    throw std::runtime_error(msg);
  }
  return {v.at(0), v.at(1)};
}

// int8 general conv (groups==1); full-IC windowed dot; mirrors Vulkan.
void q8ta_conv2d_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int weight_id = args.at(3);
  const int scales_id = args.at(5);
  const int bias_id = args.at(8);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(weight_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(scales_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_conv2d: in/weight/scales/out not tensor");
  }
  const bool has_bias =
      graph.get_value_type(bias_id) == WebGPUGraph::ValueType::Tensor;

  const int act_id = args.at(args.size() - 2);
  if (graph.get_value_type(act_id) != WebGPUGraph::ValueType::String ||
      graph.get_string(act_id) != "none") {
    throw std::runtime_error("q8ta_conv2d: only activation='none' supported");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  const auto& scales_tensor = graph.get_tensor(scales_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || weight_tensor.buffer == nullptr ||
      scales_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_conv2d: null buffer binding");
  }
  if (in_tensor.dims.size() != 4 || out_tensor.dims.size() != 4 ||
      weight_tensor.dims.size() != 2) {
    throw std::runtime_error("q8ta_conv2d: in/out must be 4D, weight 2D");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));
  const double output_scale = graph.get_double(args.at(6));
  const int output_zero_point = graph.get_int(args.at(7));

  const auto [kernel_h, kernel_w] =
      pair_or_throw(graph.get_int_list(args.at(9)), "q8ta_conv2d: kernel_size");
  const auto [stride_h, stride_w] =
      pair_or_throw(graph.get_int_list(args.at(10)), "q8ta_conv2d: stride");
  const auto [pad_h, pad_w] =
      pair_or_throw(graph.get_int_list(args.at(11)), "q8ta_conv2d: padding");
  const auto [dil_h, dil_w] =
      pair_or_throw(graph.get_int_list(args.at(12)), "q8ta_conv2d: dilation");
  const int groups = graph.get_int(args.at(13));
  if (groups != 1) {
    throw std::runtime_error("q8ta_conv2d: only groups==1 supported");
  }

  const uint64_t N = static_cast<uint64_t>(in_tensor.dims.at(0));
  const uint64_t IC = static_cast<uint64_t>(in_tensor.dims.at(1));
  const uint64_t H_in = static_cast<uint64_t>(in_tensor.dims.at(2));
  const uint64_t W_in = static_cast<uint64_t>(in_tensor.dims.at(3));
  const uint64_t OC = static_cast<uint64_t>(weight_tensor.dims.at(0));
  const uint64_t weight_row_stride =
      static_cast<uint64_t>(weight_tensor.dims.at(1));
  const uint64_t Kh = static_cast<uint64_t>(kernel_h);
  const uint64_t Kw = static_cast<uint64_t>(kernel_w);
  const uint64_t H_out = static_cast<uint64_t>(out_tensor.dims.at(2));
  const uint64_t W_out = static_cast<uint64_t>(out_tensor.dims.at(3));

  if (static_cast<uint64_t>(out_tensor.dims.at(0)) != N ||
      static_cast<uint64_t>(out_tensor.dims.at(1)) != OC) {
    throw std::runtime_error(
        "q8ta_conv2d: output must be [N, OC, H_out, W_out]");
  }
  if (weight_row_stride < Kh * Kw * IC) {
    throw std::runtime_error("q8ta_conv2d: weight row stride < Kh*Kw*IC");
  }
  const uint64_t out_numel = N * OC * H_out * W_out;
  if (out_numel == 0 || W_out % 4 != 0) {
    throw std::runtime_error(
        "q8ta_conv2d: W_out must be a nonzero multiple of 4");
  }
  const uint64_t in_numel = N * IC * H_in * W_in;
  const uint64_t weight_numel = OC * weight_row_stride;
  if (out_numel > UINT32_MAX || in_numel > UINT32_MAX ||
      weight_numel > UINT32_MAX) {
    throw std::runtime_error("q8ta_conv2d: numel exceeds u32");
  }
  if (!in_tensor.is_int8 || in_tensor.nbytes != in_numel || in_numel % 4 != 0 ||
      !weight_tensor.is_int8 || weight_tensor.nbytes != weight_numel ||
      weight_numel % 4 != 0 || !out_tensor.is_int8 ||
      out_tensor.nbytes != out_numel) {
    throw std::runtime_error("q8ta_conv2d: int8 in/weight/out size mismatch");
  }
  // scales/bias fp32 [OC]; AOT pads OC to mult-4; shader reads [0,OC) only.
  if (scales_tensor.nbytes < OC * sizeof(float)) {
    throw std::runtime_error("q8ta_conv2d: weight_scales must be fp32 [OC]");
  }
  if (has_bias) {
    const auto& b = graph.get_tensor(bias_id);
    if (b.buffer == nullptr || b.nbytes < OC * sizeof(float)) {
      throw std::runtime_error("q8ta_conv2d: bias must be fp32 [OC]");
    }
  }

  Q8taConvParams params = {};
  params.N = static_cast<uint32_t>(N);
  params.IC = static_cast<uint32_t>(IC);
  params.H_in = static_cast<uint32_t>(H_in);
  params.W_in = static_cast<uint32_t>(W_in);
  params.OC = static_cast<uint32_t>(OC);
  params.H_out = static_cast<uint32_t>(H_out);
  params.W_out = static_cast<uint32_t>(W_out);
  params.Kh = static_cast<uint32_t>(Kh);
  params.Kw = static_cast<uint32_t>(Kw);
  params.stride_h = static_cast<uint32_t>(stride_h);
  params.stride_w = static_cast<uint32_t>(stride_w);
  params.pad_h = static_cast<uint32_t>(pad_h);
  params.pad_w = static_cast<uint32_t>(pad_w);
  params.dil_h = static_cast<uint32_t>(dil_h);
  params.dil_w = static_cast<uint32_t>(dil_w);
  params.weight_row_stride = static_cast<uint32_t>(weight_row_stride);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);
  params.input_scale = static_cast<float>(input_scale);
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_output_scale = static_cast<float>(1.0 / output_scale);
  params.has_bias = has_bias ? 1u : 0u;

  const uint32_t num_words = static_cast<uint32_t>(out_numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taConv2dWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_words, wg_size, "q8ta_conv2d");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taConvParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taConvParams));

  // No-bias: bind scales as an unread placeholder (has_bias gates the read).
  WGPUBuffer bias_buf =
      has_bias ? graph.get_tensor(bias_id).buffer : scales_tensor.buffer;
  const uint64_t bias_size =
      has_bias ? graph.get_tensor(bias_id).nbytes : scales_tensor.nbytes;

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kQ8taConv2dWGSL,
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
           sizeof(Q8taConvParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "q8ta_conv2d",
       workgroup_count.y});
  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_conv2d.default, q8ta_conv2d_impl);
}

} // namespace executorch::backends::webgpu

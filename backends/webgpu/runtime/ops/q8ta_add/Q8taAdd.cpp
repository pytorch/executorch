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
#include <executorch/backends/webgpu/runtime/ops/q8ta_add/q8ta_add_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taAddParams {
  float inv_output_scale;
  float a_scale;
  float b_scale;
  float alpha;
  int32_t a_zero_point;
  int32_t b_zero_point;
  int32_t output_zero_point;
  uint32_t numel;
};
static_assert(
    sizeof(Q8taAddParams) == 32,
    "Q8taAddParams must match the WGSL Params struct (32 bytes)");

// int8 a+alpha*b then requant (CPU-golden; Vulkan glsl buffer drops alpha).
void q8ta_add_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [a, b, a_scale, a_zp, b_scale, b_zp, out_scale, out_zp, alpha, out].
  const int a_id = args.at(0);
  const int b_id = args.at(1);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(a_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(b_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_add: a/b/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& a_tensor = graph.get_tensor(a_id);
  const auto& b_tensor = graph.get_tensor(b_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (a_tensor.buffer == nullptr || b_tensor.buffer == nullptr ||
      out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_add: null buffer binding");
  }

  const double a_scale = graph.get_double(args.at(2));
  const int a_zero_point = graph.get_int(args.at(3));
  const double b_scale = graph.get_double(args.at(4));
  const int b_zero_point = graph.get_int(args.at(5));
  const double output_scale = graph.get_double(args.at(6));
  const int output_zero_point = graph.get_int(args.at(7));
  const double alpha = graph.get_double(args.at(8));

  uint64_t numel = 1;
  for (int64_t d : out_tensor.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (numel == 0 || numel % 4 != 0) {
    throw std::runtime_error("q8ta_add: numel must be a nonzero multiple of 4");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("q8ta_add: numel exceeds u32");
  }
  // All three tensors are int8 (kernel clamps to [-128,127]).
  if (!a_tensor.is_int8 || !b_tensor.is_int8 || !out_tensor.is_int8 ||
      a_tensor.nbytes != numel || b_tensor.nbytes != numel ||
      out_tensor.nbytes != numel) {
    throw std::runtime_error("q8ta_add: a/b/out must be int8 of equal numel");
  }

  Q8taAddParams params = {};
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_output_scale = static_cast<float>(1.0 / output_scale);
  params.a_scale = static_cast<float>(a_scale);
  params.b_scale = static_cast<float>(b_scale);
  params.alpha = static_cast<float>(alpha);
  params.a_zero_point = static_cast<int32_t>(a_zero_point);
  params.b_zero_point = static_cast<int32_t>(b_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);
  params.numel = static_cast<uint32_t>(numel);

  const uint32_t num_words = static_cast<uint32_t>(numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taAddWorkgroupSizeX);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, num_words, wg_size, "q8ta_add");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taAddParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taAddParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kQ8taAddWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           a_tensor.buffer,
           a_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           b_tensor.buffer,
           b_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {3, WGPUBufferBindingType_Uniform, params_buf, sizeof(Q8taAddParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "q8ta_add",
       workgroup_count.y});

  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_add.default, q8ta_add_impl);
}

} // namespace executorch::backends::webgpu

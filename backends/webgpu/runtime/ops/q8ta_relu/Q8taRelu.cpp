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
#include <executorch/backends/webgpu/runtime/ops/q8ta_relu/q8ta_relu_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

struct Q8taReluParams {
  float inv_output_scale;
  float input_scale;
  int32_t input_zero_point;
  int32_t output_zero_point;
  uint32_t numel;
  uint32_t pad0;
  uint32_t pad1;
  uint32_t pad2;
};
static_assert(
    sizeof(Q8taReluParams) == 32,
    "Q8taReluParams must match the WGSL Params struct (32 bytes)");

// int8 relu: dequant, max(x, 0), requant; mirrors Vulkan q8ta relu glsl.
void q8ta_relu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // args: [x, in_scale, in_zp, out_scale, out_zp, out]; out = args.back().
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);

  if (graph.get_value_type(in_id) != WebGPUGraph::ValueType::Tensor ||
      graph.get_value_type(out_id) != WebGPUGraph::ValueType::Tensor) {
    throw std::runtime_error("q8ta_relu: in/out is not a tensor");
  }

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("q8ta_relu: null buffer binding");
  }

  const double input_scale = graph.get_double(args.at(1));
  const int input_zero_point = graph.get_int(args.at(2));
  const double output_scale = graph.get_double(args.at(3));
  const int output_zero_point = graph.get_int(args.at(4));

  uint64_t numel = 1;
  for (int64_t d : out_tensor.dims) {
    numel *= static_cast<uint64_t>(d);
  }
  if (numel == 0 || numel % 4 != 0) {
    throw std::runtime_error(
        "q8ta_relu: numel must be a nonzero multiple of 4");
  }
  if (numel > UINT32_MAX) {
    throw std::runtime_error("q8ta_relu: numel exceeds u32");
  }
  // in/out int8 (kernel clamps to [-128,127]) of equal numel.
  if (!in_tensor.is_int8 || !out_tensor.is_int8 || in_tensor.nbytes != numel ||
      out_tensor.nbytes != numel) {
    throw std::runtime_error("q8ta_relu: in/out must be int8 of equal numel");
  }

  Q8taReluParams params = {};
  // Reciprocal in double then cast, matching torch's f32(1.0 / f64(scale)).
  params.inv_output_scale = static_cast<float>(1.0 / output_scale);
  params.input_scale = static_cast<float>(input_scale);
  params.input_zero_point = static_cast<int32_t>(input_zero_point);
  params.output_zero_point = static_cast<int32_t>(output_zero_point);
  params.numel = static_cast<uint32_t>(numel);

  const uint32_t num_words = static_cast<uint32_t>(numel / 4);
  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kQ8taReluWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_words, wg_size, "q8ta_relu");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  WGPUBuffer params_buf =
      utils::make_uniform(device, &params, sizeof(Q8taReluParams));
  graph.add_uniform_buffer_bytes(sizeof(Q8taReluParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kQ8taReluWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in_tensor.buffer,
           in_tensor.nbytes},
          {1,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           params_buf,
           sizeof(Q8taReluParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "q8ta_relu",
       workgroup_count.y});

  // Dynamic shapes (supports_resize): recompute numel + dispatch, rewrite UBO.
  Q8taReluParams base = params;
  WGPUBuffer p_buf = params_buf;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, base, wg_size, dispatch_idx, p_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        uint64_t n = utils::numel_of(d);
        if (n == 0 || n % 4 != 0 || n > UINT32_MAX) {
          throw std::runtime_error(
              "q8ta_relu(resize): numel must be a u32 "
              "nonzero multiple of 4");
        }
        Q8taReluParams p = base;
        p.numel = static_cast<uint32_t>(n);
        wgpuQueueWriteBuffer(g.queue(), p_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(),
            static_cast<uint32_t>(n / 4),
            wg_size,
            "q8ta_relu(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
        g.set_cur_dims(out_id, d);
      });

  graph.own_uniform_buffer(params_buf);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.q8ta_relu.default, q8ta_relu_impl);
}

} // namespace executorch::backends::webgpu

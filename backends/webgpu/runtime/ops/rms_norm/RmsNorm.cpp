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
#include <executorch/backends/webgpu/runtime/ops/rms_norm/RmsNormFusion.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_vec4_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/rms_norm/rms_norm_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching the WGSL Params struct (16-byte aligned).
struct RmsNormParams {
  uint32_t num_rows;
  uint32_t row_width;
  float epsilon;
  uint32_t _pad;
};
static_assert(sizeof(RmsNormParams) == 16, "RmsNormParams must be 16 bytes");

// Resize hook body: recompute num_rows + rewrite the UBO for the live input.
void resize_rms_norm(
    WebGPUGraph& g,
    int in_id,
    int out_id,
    uint32_t row_width,
    float epsilon,
    size_t dispatch_idx,
    WGPUBuffer params_buf) {
  const auto& d = g.cur_dims(in_id);
  const uint64_t numel = utils::numel_of(d);
  if (numel % static_cast<uint64_t>(row_width) != 0) {
    throw std::runtime_error(
        "WebGPU rms_norm: numel not a multiple of row_width");
  }
  const uint32_t rows =
      static_cast<uint32_t>(numel / static_cast<uint64_t>(row_width));
  if (rows == 0) {
    throw std::runtime_error("WebGPU rms_norm: zero rows");
  }
  RmsNormParams p = {};
  p.num_rows = rows;
  p.row_width = row_width;
  p.epsilon = epsilon;
  wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
  const utils::WgCount wg =
      utils::compute_2d_workgroup_count(g.device(), rows, 1, "rms_norm");
  g.dispatch_at(dispatch_idx).workgroup_count_x = wg.x;
  g.dispatch_at(dispatch_idx).workgroup_count_y = wg.y;
  g.set_cur_dims(out_id, d);
}

void rms_norm_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // et_vk.rms_norm.default args: [in, weight, eps, out]
  const int in_id = args.at(0);
  const int weight_id = args.at(1);
  const int eps_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  // Get epsilon (Double from a Python float; defaults to float32 eps)
  float epsilon = std::numeric_limits<float>::epsilon();
  if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Double) {
    epsilon = static_cast<float>(graph.get_double(eps_id));
  } else if (graph.get_value_type(eps_id) == WebGPUGraph::ValueType::Int) {
    epsilon = static_cast<float>(graph.get_int(eps_id));
  }

  // row_width = last dim; num_rows = product of the rest (PyTorch NCHW order)
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);
  if (in_tensor.dims.empty() || in_tensor.nbytes == 0) {
    throw std::runtime_error("WebGPU rms_norm: empty input");
  }
  if (in_tensor.dims.back() <= 0 ||
      static_cast<uint64_t>(in_tensor.dims.back()) > UINT32_MAX) {
    throw std::runtime_error("WebGPU rms_norm: invalid row width");
  }
  const uint32_t row_width = static_cast<uint32_t>(in_tensor.dims.back());
  const uint64_t in_numel = utils::numel_of(in_tensor.dims);
  if (!utils::is_fp32_tensor(in_tensor) || !utils::is_fp32_tensor(out_tensor) ||
      !utils::is_fp32_tensor(weight_tensor) ||
      out_tensor.dims != in_tensor.dims ||
      utils::numel_of(weight_tensor.dims) != row_width) {
    throw std::runtime_error(
        "WebGPU rms_norm: expected fp32 input/output and row-width weight");
  }
  if (in_numel % row_width != 0u || in_numel / row_width == 0u ||
      in_numel / row_width > UINT32_MAX) {
    throw std::runtime_error("WebGPU rms_norm: invalid row count");
  }
  const uint32_t num_rows = static_cast<uint32_t>(in_numel / row_width);
  // Rows can exceed the per-dim grid cap (QK-norm at prefill), so fold x/y.
  const utils::WgCount wg_count =
      utils::compute_2d_workgroup_count(device, num_rows, 1, "rms_norm");

  // Create uniform buffer for params
  RmsNormParams params = {};
  params.num_rows = num_rows;
  params.row_width = row_width;
  params.epsilon = epsilon;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(RmsNormParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped =
      wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(RmsNormParams));
  std::memcpy(mapped, &params, sizeof(RmsNormParams));
  wgpuBufferUnmap(uniform_buffer);

  graph.add_uniform_buffer_bytes(sizeof(RmsNormParams));

  // Select the vec4 kernel when the row width is a multiple of 4 (every Llama
  // hidden size qualifies); fall back to the scalar kernel otherwise. The two
  // kernels are equivalent up to floating-point reassociation (the vec4
  // reduction reorders the sum, so not bit-identical) and share the same bind
  // group + dispatch.
  const bool use_vec4 = (row_width % 4u == 0u);

  const char* shader_src = use_vec4 ? kRmsNormVec4WGSL : kRmsNormWGSL;

  // Runtime-overridable workgroup size (mirrors add op); clamp only reduces.
  // Pow2 required: the kernel halves the reduction stride (wg_size / 2u).
  const uint32_t wg_size =
      utils::clamp_workgroup_size_pow2(device, kRmsNormWorkgroupSizeX);
  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      shader_src,
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
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(RmsNormParams)},
      },
      &wg_size_constant,
      1);

  // One workgroup per row (kRmsNormWorkgroupSizeX threads cooperate per row)
  static_assert(
      kRmsNormWorkgroupSizeX == 64,
      "kRmsNormWorkgroupSizeX must match override wg_size default (64)");
  static_assert(
      kRmsNormVec4WorkgroupSizeX == 64,
      "kRmsNormVec4WorkgroupSizeX must match override wg_size default (64)");
  const size_t dispatch_idx = graph.add_dispatch(
      {bundle.pipeline, bundle.bind_group, wg_count.x, "rms_norm", wg_count.y});

  // Offer this dispatch to the add/mul merge; vec4 at 64-wide only.
  if (use_vec4 && wg_size == kRmsNormVec4WorkgroupSizeX) {
    fusion::record_rms_norm(
        graph,
        in_id,
        weight_id,
        out_id,
        num_rows,
        row_width,
        dispatch_idx,
        uniform_buffer,
        bundle.bind_group);
  } else {
    fusion::invalidate_record(graph);
  }

  // Dynamic shapes: recompute num_rows + rewrite the UBO for the live input.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, row_width, epsilon, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        resize_rms_norm(
            g, in_id, out_id, row_width, epsilon, dispatch_idx, params_buf);
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.rms_norm.default, rms_norm_impl);
}

} // namespace executorch::backends::webgpu

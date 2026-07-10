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
  if (rows > 65535u) {
    throw std::runtime_error(
        "WebGPU rms_norm: num_rows exceeds the 1D dispatch limit (65535)");
  }
  RmsNormParams p = {};
  p.num_rows = rows;
  p.row_width = row_width;
  p.epsilon = epsilon;
  wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
  g.dispatch_at(dispatch_idx).workgroup_count_x = rows;
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
  if (in_tensor.dims.empty() || in_tensor.nbytes == 0) {
    throw std::runtime_error("WebGPU rms_norm: empty input");
  }
  const uint32_t row_width = static_cast<uint32_t>(in_tensor.dims.back());
  if (row_width == 0) {
    throw std::runtime_error("WebGPU rms_norm: zero row width");
  }
  uint64_t in_numel = 1;
  for (int64_t d : in_tensor.dims) {
    in_numel *= static_cast<uint64_t>(d);
  }
  // fp32-only shader: bail if the bytes don't match an fp32 element count.
  if (in_tensor.nbytes != in_numel * sizeof(float)) {
    throw std::runtime_error("WebGPU rms_norm: fp32-only (byte-size mismatch)");
  }
  const uint32_t num_rows = static_cast<uint32_t>(in_numel / row_width);
  if (num_rows == 0) {
    throw std::runtime_error("WebGPU rms_norm: zero rows");
  }
  // Validate the 1D dispatch limit before allocating any GPU objects.
  if (num_rows > 65535u) {
    throw std::runtime_error(
        "WebGPU rms_norm: num_rows exceeds the 1D dispatch limit (65535)");
  }

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

  // Bind group buffers: out (rw) + in/weight (ro storage) + params.
  const auto& out_tensor = graph.get_tensor(out_id);
  const auto& weight_tensor = graph.get_tensor(weight_id);

  // Pass the selected vec4/scalar WGSL string; no override constant.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      use_vec4 ? kRmsNormVec4WGSL : kRmsNormWGSL,
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
      });

  // One workgroup per row (kRmsNormWorkgroupSizeX threads cooperate per row)
  static_assert(
      kRmsNormWorkgroupSizeX == 64,
      "must match @workgroup_size and WG_SIZE in rms_norm.wgsl");
  static_assert(
      kRmsNormVec4WorkgroupSizeX == 64,
      "must match @workgroup_size and WG_SIZE in rms_norm_vec4.wgsl");
  const size_t dispatch_idx =
      graph.add_dispatch({bundle.pipeline, bundle.bind_group, num_rows});

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

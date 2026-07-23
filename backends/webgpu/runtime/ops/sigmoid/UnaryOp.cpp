/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUShaderRegistry.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <webgpu/webgpu.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct; 16-byte aligned.
struct UnaryParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

void resize_unary(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    uint32_t wg_size,
    size_t dispatch_idx,
    WGPUBuffer params_buffer) {
  const auto& dims = graph.cur_dims(in_id);
  const uint64_t numel = utils::numel_of(dims);
  graph.set_cur_dims(out_id, dims);
  UnaryParams params = {};
  params.num_elements = static_cast<uint32_t>(numel);
  wgpuQueueWriteBuffer(
      graph.queue(), params_buffer, 0, &params, sizeof(params));
  graph.dispatch_at(dispatch_idx).workgroup_count_x =
      utils::compute_1d_workgroup_count(
          graph.device(),
          static_cast<uint32_t>(numel),
          wg_size,
          "unary(resize)");
}

// Generic elementwise unary op; mirrors Vulkan add_unary_op_node (UnaryOp.cpp).
void add_unary_op(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    const char* shader_name,
    const char* op_name) {
  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error(std::string(op_name) + ": null buffer binding");
  }
  if (in_tensor.nbytes % sizeof(float) != 0 ||
      out_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error(
        std::string(op_name) + ": operand not 4-byte aligned");
  }
  if (in_tensor.nbytes != out_tensor.nbytes) {
    throw std::runtime_error(
        std::string(op_name) + ": input/output size mismatch");
  }

  const uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));
  const uint32_t wg_size = utils::clamp_workgroup_size(
      device, get_webgpu_shader_info(shader_name).workgroup_size_x);
  const uint32_t workgroup_count =
      utils::compute_1d_workgroup_count(device, num_elements, wg_size, op_name);

  UnaryParams params = {};
  params.num_elements = num_elements;
  WGPUBuffer params_buffer = graph.create_params_buffer(params);

  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = shader_name;
  descriptor.kernel_name = op_name;
  descriptor.bindings = {
      {in_tensor.buffer, 0u, in_tensor.nbytes},
      {out_tensor.buffer, 0u, out_tensor.nbytes},
      {params_buffer, 0u, sizeof(UnaryParams)}};
  descriptor.constants = {{"wg_size", static_cast<double>(wg_size)}};
  descriptor.grid = {workgroup_count, 1u};
  const size_t dispatch_idx = graph.add_compute_dispatch(descriptor);

  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, wg_size, dispatch_idx, params_buffer](WebGPUGraph& g) {
        resize_unary(g, in_id, out_id, wg_size, dispatch_idx, params_buffer);
      });
}

void sigmoid_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.sigmoid.default args: [in, out]
  add_unary_op(graph, args.at(0), args.at(1), "sigmoid", "sigmoid");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sigmoid.default, sigmoid_impl);
}

} // namespace executorch::backends::webgpu

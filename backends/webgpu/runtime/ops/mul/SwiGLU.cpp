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

#include <limits>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform layout matching silu_mul_fused.wgsl's Params struct.
struct SiluMulParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

uint32_t checked_silu_mul_numel(const std::vector<int64_t>& dims) {
  const uint64_t numel = utils::numel_of(dims);
  if (numel == 0 || numel > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("silu_mul_fused: element count out of range");
  }
  return static_cast<uint32_t>(numel);
}

void resize_silu_mul_fused(
    WebGPUGraph& graph,
    int gate_id,
    int up_id,
    int out_id,
    uint32_t wg_size,
    size_t dispatch_idx,
    WGPUBuffer params_buffer) {
  const auto& gate_dims = graph.cur_dims(gate_id);
  const auto& up_dims = graph.cur_dims(up_id);
  if (gate_dims != up_dims) {
    throw std::runtime_error("silu_mul_fused(resize): gate/up shape mismatch");
  }
  const uint32_t live_numel = checked_silu_mul_numel(gate_dims);
  const size_t live_nbytes = static_cast<size_t>(live_numel) * sizeof(float);
  if (graph.get_tensor(gate_id).cur_nbytes != live_nbytes ||
      graph.get_tensor(up_id).cur_nbytes != live_nbytes) {
    throw std::runtime_error(
        "silu_mul_fused(resize): gate/up byte-size mismatch");
  }
  graph.set_cur_dims(out_id, gate_dims);
  const SiluMulParams params = {live_numel, {0u, 0u, 0u}};
  wgpuQueueWriteBuffer(
      graph.queue(), params_buffer, 0, &params, sizeof(params));
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      graph.device(), live_numel, wg_size, "silu_mul_fused(resize)");
  auto& dispatch = graph.dispatch_at(dispatch_idx);
  dispatch.workgroup_count_x = workgroup_count.x;
  dispatch.workgroup_count_y = workgroup_count.y;
}

void swiglu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // et_vk.swiglu.default args: [gate, up, out]
  const int gate_id = args.at(0);
  const int up_id = args.at(1);
  const int out_id = args.at(2);

  const auto& gate = graph.get_tensor(gate_id);
  const auto& up = graph.get_tensor(up_id);
  const auto& out = graph.get_tensor(out_id);
  if (!utils::is_fp32_tensor(gate) || !utils::is_fp32_tensor(up) ||
      !utils::is_fp32_tensor(out)) {
    throw std::runtime_error("swiglu: only fp32 is supported");
  }
  if (gate.dims != up.dims || gate.dims != out.dims) {
    throw std::runtime_error("swiglu: gate, up, and out shapes must match");
  }

  const uint32_t num_elements = checked_silu_mul_numel(gate.dims);
  const uint32_t wg_size = utils::clamp_workgroup_size(
      graph.device(),
      get_webgpu_shader_info("silu_mul_fused").workgroup_size_x);
  const utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      graph.device(), num_elements, wg_size, "silu_mul_fused");

  const SiluMulParams params = {num_elements, {0u, 0u, 0u}};
  WGPUBuffer params_buffer = graph.create_params_buffer(params);
  WebGPUComputeDispatchDescriptor descriptor;
  descriptor.shader_name = "silu_mul_fused";
  descriptor.bindings = {
      {gate.buffer, 0u, gate.nbytes},
      {up.buffer, 0u, up.nbytes},
      {out.buffer, 0u, out.nbytes},
      {params_buffer, 0u, sizeof(SiluMulParams)}};
  descriptor.constants = {{"wg_size", static_cast<double>(wg_size)}};
  descriptor.grid = {workgroup_count.x, workgroup_count.y};
  const size_t dispatch_idx = graph.add_compute_dispatch(descriptor);

  auto resize = [gate_id, up_id, out_id, wg_size, dispatch_idx, params_buffer](
                    WebGPUGraph& g) {
    resize_silu_mul_fused(
        g, gate_id, up_id, out_id, wg_size, dispatch_idx, params_buffer);
  };
  graph.add_tensor_resize_hook(gate_id, resize);
  graph.add_tensor_resize_hook(up_id, resize);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(et_vk.swiglu.default, swiglu_impl);
}

} // namespace executorch::backends::webgpu

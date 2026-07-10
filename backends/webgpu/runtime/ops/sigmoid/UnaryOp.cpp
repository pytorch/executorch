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
#include <executorch/backends/webgpu/runtime/ops/relu/relu_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/sigmoid/sigmoid_wgsl.h>

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

// Generic elementwise unary op; mirrors Vulkan add_unary_op_node (UnaryOp.cpp).
void add_unary_op(
    WebGPUGraph& graph,
    int in_id,
    int out_id,
    const char* wgsl_source,
    uint32_t wg_size_x,
    const char* op_name) {
  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  // 4-byte (fp32) alignment guard on both operands; also the dtype guard.
  utils::check_elementwise_fp32_io(in_tensor, out_tensor, op_name);

  uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  // Adaptive 1D->2D dispatch: wg=clamp(device,256) + 2D-spill past the 65535
  // per-dim ceiling. The shader decodes idx via num_workgroups.x, so the live
  // count_x sets the stride at runtime (resize-safe, no override to re-bake).
  uint32_t wg_size = utils::clamp_workgroup_size(device, wg_size_x);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, num_elements, wg_size, op_name);

  WGPUConstantEntry wg_size_constant = utils::make_wg_size_constant(wg_size);

  UnaryParams params = {};
  params.num_elements = num_elements;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(UnaryParams));
  graph.add_uniform_buffer_bytes(sizeof(UnaryParams));

  // input (read storage) + output (storage) + params.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      wgsl_source,
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
           uniform_buffer,
           sizeof(UnaryParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_idx = graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, workgroup_count.x, workgroup_count.y);

  // Dynamic shapes: recompute num_elements/dispatch for the live shape.
  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& d = g.cur_dims(in_id);
        const uint64_t numel = utils::numel_of(d);
        g.set_cur_dims(out_id, d);
        UnaryParams p = {};
        p.num_elements = static_cast<uint32_t>(numel);
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), static_cast<uint32_t>(numel), wg_size, "unary(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      });

  // Graph owns it so the resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

void sigmoid_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.sigmoid.default args: [in, out]
  add_unary_op(
      graph,
      args.at(0),
      args.at(1),
      kSigmoidWGSL,
      kSigmoidWorkgroupSizeX,
      "sigmoid");
}

void relu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.relu.default args: [in, out]
  add_unary_op(
      graph, args.at(0), args.at(1), kReluWGSL, kReluWorkgroupSizeX, "relu");
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.sigmoid.default, sigmoid_impl);
  WEBGPU_REGISTER_OP(aten.relu.default, relu_impl);
}

} // namespace executorch::backends::webgpu

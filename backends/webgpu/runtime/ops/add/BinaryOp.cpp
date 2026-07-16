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
#include <executorch/backends/webgpu/runtime/ops/add/binary_add_wgsl.h>

#include <webgpu/webgpu.h>

#include <cmath>
#include <cstring>

namespace executorch {
namespace backends {
namespace webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct.
// Must be 16-byte aligned for WebGPU uniform buffer requirements.
struct AddParams {
  uint32_t num_elements;
  float alpha;
  uint32_t _pad[2]; // pad to 16 bytes
};

void add_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  // aten.add.Tensor args: [in1, in2, alpha, out]
  const int in1_id = args.at(0);
  const int in2_id = args.at(1);
  const int alpha_id = args.at(2);
  const int out_id = args.at(3);

  WGPUDevice device = graph.device();

  // Get alpha value (defaults to 1.0 if not a scalar)
  float alpha = 1.0f;
  if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Int) {
    alpha = static_cast<float>(graph.get_int(alpha_id));
  } else if (graph.get_value_type(alpha_id) == WebGPUGraph::ValueType::Double) {
    alpha = static_cast<float>(graph.get_double(alpha_id));
  }

  const auto& out_tensor = graph.get_tensor(out_id);
  uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  uint32_t wg_size =
      utils::clamp_workgroup_size(device, kBinaryAddWorkgroupSizeX);
  utils::WgCount workgroup_count =
      utils::compute_2d_workgroup_count(device, num_elements, wg_size, "add");

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  // Create uniform buffer for params
  AddParams params = {};
  params.num_elements = num_elements;
  params.alpha = alpha;

  WGPUBufferDescriptor uniform_desc = {};
  uniform_desc.size = sizeof(AddParams);
  uniform_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
  uniform_desc.mappedAtCreation = true;
  WGPUBuffer uniform_buffer = wgpuDeviceCreateBuffer(device, &uniform_desc);
  void* mapped = wgpuBufferGetMappedRange(uniform_buffer, 0, sizeof(AddParams));
  std::memcpy(mapped, &params, sizeof(AddParams));
  wgpuBufferUnmap(uniform_buffer);

  graph.add_uniform_buffer_bytes(sizeof(AddParams));

  // Create bind group with actual buffers
  const auto& in1_tensor = graph.get_tensor(in1_id);
  const auto& in2_tensor = graph.get_tensor(in2_id);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kBinaryAddWGSL,
      {
          {0,
           WGPUBufferBindingType_ReadOnlyStorage,
           in1_tensor.buffer,
           in1_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           in2_tensor.buffer,
           in2_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Storage,
           out_tensor.buffer,
           out_tensor.nbytes},
          {3, WGPUBufferBindingType_Uniform, uniform_buffer, sizeof(AddParams)},
      },
      &wg_size_constant,
      1);

  graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       workgroup_count.x,
       "",
       workgroup_count.y});
  const size_t dispatch_idx = graph.num_dispatches() - 1;

  // Dynamic shapes: recompute numel/dispatch; out follows the larger operand.
  WGPUBuffer params_buf = uniform_buffer;
  auto add_resize =
      [in1_id, in2_id, out_id, alpha, wg_size, dispatch_idx, params_buf](
          WebGPUGraph& g) {
        const auto& d1 = g.cur_dims(in1_id);
        const auto& d2 = g.cur_dims(in2_id);
        const uint64_t n1 = utils::numel_of(d1);
        const uint64_t n2 = utils::numel_of(d2);
        const uint64_t numel = n2 > n1 ? n2 : n1;
        const uint64_t n_min = n2 > n1 ? n1 : n2;
        // The flat add follows the larger operand and broadcasts the smaller;
        // valid only when the smaller tiles evenly into it (rejects e.g. [4,1]
        // vs [1,3], whose true [4,3] result this flat kernel cannot produce).
        if (n_min == 0u || numel % n_min != 0u) {
          throw std::runtime_error(
              "add(resize): operands are not broadcast-compatible by numel");
        }
        g.set_cur_dims(out_id, n2 > n1 ? d2 : d1);
        AddParams p = {};
        p.num_elements = static_cast<uint32_t>(numel);
        p.alpha = alpha;
        wgpuQueueWriteBuffer(g.queue(), params_buf, 0, &p, sizeof(p));
        const utils::WgCount wgc = utils::compute_2d_workgroup_count(
            g.device(), static_cast<uint32_t>(numel), wg_size, "add(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x = wgc.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y = wgc.y;
      };
  graph.add_tensor_resize_hook(in1_id, add_resize);
  graph.add_tensor_resize_hook(in2_id, add_resize);

  // Graph owns it so a resize hook can rewrite it; freed in the dtor.
  graph.own_uniform_buffer(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.add.Tensor, add_impl);
}

} // namespace webgpu
} // namespace backends
} // namespace executorch

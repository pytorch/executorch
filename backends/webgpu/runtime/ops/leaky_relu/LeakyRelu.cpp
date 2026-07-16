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
#include <executorch/backends/webgpu/runtime/ops/leaky_relu/leaky_relu_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

struct LeakyReluParams {
  uint32_t num_elements;
  float neg_slope;
  uint32_t _pad[2];
};

// aten.leaky_relu.default: fp32 elementwise; mirrors Vulkan leaky_relu.
void leaky_relu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() < 2) {
    throw std::runtime_error("WebGPU leaky_relu: expected >=2 args");
  }
  const int in_id = args.at(0);
  const int out_id = args.at(args.size() - 1);
  const float neg_slope =
      (args.size() >= 3) ? utils::scalar_or(graph, args.at(1), 0.01f) : 0.01f;

  WGPUDevice device = graph.device();
  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  if (in_tensor.buffer == nullptr || out_tensor.buffer == nullptr) {
    throw std::runtime_error("leaky_relu: null buffer binding");
  }
  if (in_tensor.nbytes != out_tensor.nbytes ||
      out_tensor.nbytes % sizeof(float) != 0) {
    throw std::runtime_error("leaky_relu: fp32-only / size mismatch");
  }
  const uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  utils::DispatchGrid grid = utils::compute_dispatch_grid(
      device, num_elements, kLeakyReluWorkgroupSizeX, "leaky_relu");

  auto constants = utils::make_grid_constants(grid);

  LeakyReluParams params = {};
  params.num_elements = num_elements;
  params.neg_slope = neg_slope;
  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(LeakyReluParams));
  graph.add_uniform_buffer_bytes(sizeof(LeakyReluParams));

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kLeakyReluWGSL,
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
           sizeof(LeakyReluParams)},
      },
      constants.data(),
      constants.size());

  WebGPUDispatch dispatch{};
  dispatch.pipeline = bundle.pipeline;
  dispatch.bind_group = bundle.bind_group;
  dispatch.workgroup_count_x = grid.count_x;
  dispatch.workgroup_count_y = grid.count_y;
  graph.add_dispatch(dispatch);

  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.leaky_relu.default, leaky_relu_impl);
}

} // namespace executorch::backends::webgpu

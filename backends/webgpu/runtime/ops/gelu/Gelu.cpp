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
#include <executorch/backends/webgpu/runtime/ops/gelu/gelu_wgsl.h>

#include <webgpu/webgpu.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

// Uniform buffer layout matching the WGSL Params struct; 16-byte aligned.
struct GeluParams {
  uint32_t num_elements;
  uint32_t _pad[3];
};

// aten.gelu.default args: [in, approximate, out] (mirrors Vulkan UnaryOp.cpp
// gelu — args[1] is the `approximate` string). approximate="none" selects the
// exact (erf) entry point; anything else (e.g. "tanh") selects the tanh
// approximation entry point.
void gelu_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int in_id = args.at(0);
  const int out_id = args.at(2);
  const bool exact = graph.get_string(args.at(1)) == "none";

  WGPUDevice device = graph.device();

  const auto& in_tensor = graph.get_tensor(in_id);
  const auto& out_tensor = graph.get_tensor(out_id);
  utils::check_elementwise_fp32_io(in_tensor, out_tensor, "gelu");

  const uint64_t num_elements64 = out_tensor.nbytes / sizeof(float);
  if (num_elements64 >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error(
        "WebGPU gelu: element count exceeds the flattened 2D dispatch limit");
  }
  const uint32_t num_elements = static_cast<uint32_t>(num_elements64);

  // Each thread handles up to 4 elements (vec4 body + scalar-tail idiom).
  uint32_t num_vec4_threads = utils::div_up(num_elements, 4u);
  uint32_t wg_size = utils::clamp_workgroup_size(device, kGeluWorkgroupSizeX);
  utils::WgCount workgroup_count = utils::compute_2d_workgroup_count(
      device, num_vec4_threads, wg_size, "gelu");

  WGPUConstantEntry wg_constant = utils::make_wg_size_constant(wg_size);

  GeluParams params = {};
  params.num_elements = num_elements;

  WGPUBuffer uniform_buffer = graph.create_params_buffer(params);

  // input (read storage) + output (storage) + params. The exact/approximate
  // choice is baked into the compiled pipeline via the entry point (mirrors
  // onnxruntime's WebGPU EP), not a per-invocation select() — `main_tanh`/
  // `main_erf` each contain only their own formula, no double-eval.
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kGeluWGSL,
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
           sizeof(GeluParams)},
      },
      &wg_constant,
      1,
      exact ? "main_erf" : "main_tanh");

  const size_t dispatch_idx = graph.add_dispatch_2d(
      bundle.pipeline, bundle.bind_group, workgroup_count.x, workgroup_count.y);

  WGPUBuffer params_buf = uniform_buffer;
  graph.add_tensor_resize_hook(
      in_id,
      [in_id, out_id, wg_size, dispatch_idx, params_buf](WebGPUGraph& g) {
        const auto& dims = g.cur_dims(in_id);
        const uint64_t num_elements64 = utils::numel_of(dims);
        if (num_elements64 >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          throw std::runtime_error(
              "WebGPU gelu(resize): element count exceeds the flattened 2D "
              "dispatch limit");
        }
        const uint32_t num_elements = static_cast<uint32_t>(num_elements64);
        g.set_cur_dims(out_id, dims);

        GeluParams params = {};
        params.num_elements = num_elements;
        wgpuQueueWriteBuffer(
            g.queue(), params_buf, 0, &params, sizeof(GeluParams));

        const uint32_t num_vec4_threads = utils::div_up(num_elements, 4u);
        const utils::WgCount resized_workgroup_count =
            utils::compute_2d_workgroup_count(
                g.device(), num_vec4_threads, wg_size, "gelu(resize)");
        g.dispatch_at(dispatch_idx).workgroup_count_x =
            resized_workgroup_count.x;
        g.dispatch_at(dispatch_idx).workgroup_count_y =
            resized_workgroup_count.y;
      });
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.gelu.default, gelu_impl);
}

} // namespace executorch::backends::webgpu

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

  uint32_t num_elements =
      static_cast<uint32_t>(out_tensor.nbytes / sizeof(float));

  // Each thread handles up to 4 elements (vec4 body + scalar-tail idiom).
  uint32_t num_vec4_threads = utils::div_up(num_elements, 4u);
  uint32_t wg_size = utils::clamp_workgroup_size(device, kGeluWorkgroupSizeX);
  uint32_t workgroup_count = utils::compute_1d_workgroup_count(
      device, num_vec4_threads, wg_size, "gelu");

  WGPUConstantEntry wg_constant = utils::make_wg_size_constant(wg_size);

  GeluParams params = {};
  params.num_elements = num_elements;

  WGPUBuffer uniform_buffer =
      utils::make_uniform(device, &params, sizeof(GeluParams));
  graph.add_uniform_buffer_bytes(sizeof(GeluParams));

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

  graph.add_dispatch({bundle.pipeline, bundle.bind_group, workgroup_count});

  // Drop our ref; the bind group keeps the uniform buffer alive until release.
  wgpuBufferRelease(uniform_buffer);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.gelu.default, gelu_impl);
}

} // namespace executorch::backends::webgpu

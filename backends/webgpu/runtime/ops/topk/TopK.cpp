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
#include <executorch/backends/webgpu/runtime/ops/topk/topk_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

constexpr int64_t kInputWidth = 2048;
constexpr int64_t kOutputWidth = 32;

void topk_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  if (args.size() != 6u) {
    throw std::runtime_error("WebGPU topk: malformed argument list");
  }
  const int input_id = args[0];
  const int k_id = args[1];
  const int dim_id = args[2];
  const int largest_id = args[3];
  const int sorted_id = args[4];
  const int output_list_id = args[5];

  if (graph.get_value_type(k_id) != WebGPUGraph::ValueType::Int ||
      graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int ||
      graph.get_value_type(largest_id) != WebGPUGraph::ValueType::Bool ||
      graph.get_value_type(sorted_id) != WebGPUGraph::ValueType::Bool ||
      graph.get_value_type(output_list_id) !=
          WebGPUGraph::ValueType::ValueList) {
    throw std::runtime_error("WebGPU topk: malformed scalar arguments");
  }
  if (graph.get_int(k_id) != kOutputWidth || graph.get_int(dim_id) != -1 ||
      !graph.get_bool(largest_id) || !graph.get_bool(sorted_id)) {
    throw std::runtime_error(
        "WebGPU topk: requires k=32, dim=-1, largest=true, sorted=true");
  }

  const auto& output_ids = graph.get_value_list(output_list_id);
  if (output_ids.size() != 2u) {
    throw std::runtime_error(
        "WebGPU topk: expected values and indices outputs");
  }
  const auto& input = graph.get_tensor(input_id);
  const auto& values = graph.get_tensor(output_ids[0]);
  const auto& indices = graph.get_tensor(output_ids[1]);
  const std::vector<int64_t> input_dims = {1, 1, kInputWidth};
  const std::vector<int64_t> output_dims = {1, 1, kOutputWidth};
  if (input.buffer == nullptr || input.is_int ||
      input.elem_size != sizeof(float) || input.dims != input_dims ||
      input.nbytes != kInputWidth * sizeof(float)) {
    throw std::runtime_error("WebGPU topk: input must be fp32 [1,1,2048]");
  }
  if (values.buffer == nullptr || values.is_int ||
      values.elem_size != sizeof(float) || values.dims != output_dims ||
      values.nbytes != kOutputWidth * sizeof(float)) {
    throw std::runtime_error("WebGPU topk: values must be fp32 [1,1,32]");
  }
  if (indices.buffer == nullptr || !indices.is_int ||
      indices.elem_size != sizeof(int32_t) || indices.dims != output_dims ||
      indices.nbytes != kOutputWidth * sizeof(int32_t)) {
    throw std::runtime_error(
        "WebGPU topk: indices must use effective i32 [1,1,32] storage");
  }

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      graph.device(),
      kTopkWGSL,
      {
          {0, WGPUBufferBindingType_Storage, values.buffer, values.nbytes},
          {1, WGPUBufferBindingType_Storage, indices.buffer, indices.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           input.buffer,
           input.nbytes},
      });
  try {
    graph.add_dispatch(
        {bundle.pipeline, bundle.bind_group, 1u, "topk_staged_serial"});
  } catch (...) {
    wgpuComputePipelineRelease(bundle.pipeline);
    wgpuBindGroupRelease(bundle.bind_group);
    throw;
  }
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.topk.default, topk_impl);
}

} // namespace executorch::backends::webgpu

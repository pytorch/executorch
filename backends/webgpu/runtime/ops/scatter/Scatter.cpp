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
#include <executorch/backends/webgpu/runtime/ops/scatter/scatter_unique_indices_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/scatter/scatter_wgsl.h>
#include <executorch/backends/webgpu/runtime/ops/view_copy/view_copy.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace executorch::backends::webgpu {

namespace {

constexpr int64_t kVocabSize = 262144;
constexpr int64_t kSelectedCount = 4096;

void scatter_impl(
    WebGPUGraph& graph,
    const std::vector<int>& args,
    bool unique_indices) {
  if (args.size() != 5u) {
    throw std::runtime_error("WebGPU scatter: malformed argument list");
  }
  const int input_id = args[0];
  const int dim_id = args[1];
  const int index_id = args[2];
  const int source_id = args[3];
  const int output_id = args[4];

  if (graph.get_value_type(dim_id) != WebGPUGraph::ValueType::Int ||
      graph.get_int(dim_id) != -1) {
    throw std::runtime_error("WebGPU scatter: requires dim=-1");
  }
  const auto& input = graph.get_tensor(input_id);
  const auto& index = graph.get_tensor(index_id);
  const auto& source = graph.get_tensor(source_id);
  const auto& output = graph.get_tensor(output_id);
  const std::vector<int64_t> vocab_dims = {1, 1, kVocabSize};
  const std::vector<int64_t> selected_dims = {1, 1, kSelectedCount};
  if (input.buffer == nullptr || input.is_int ||
      input.elem_size != sizeof(float) || input.dims != vocab_dims ||
      input.nbytes != kVocabSize * sizeof(float)) {
    throw std::runtime_error("WebGPU scatter: input must be fp32 [1,1,262144]");
  }
  if (index.buffer == nullptr || !index.is_int ||
      index.elem_size != sizeof(int32_t) || index.dims != selected_dims ||
      index.nbytes != kSelectedCount * sizeof(int32_t)) {
    throw std::runtime_error(
        "WebGPU scatter: index must use effective i32 [1,1,4096] storage");
  }
  if (source.buffer == nullptr || source.is_int ||
      source.elem_size != sizeof(float) || source.dims != selected_dims ||
      source.nbytes != kSelectedCount * sizeof(float)) {
    throw std::runtime_error("WebGPU scatter: source must be fp32 [1,1,4096]");
  }
  if (output.buffer == nullptr || output.is_int ||
      output.elem_size != sizeof(float) || output.dims != vocab_dims ||
      output.nbytes != kVocabSize * sizeof(float)) {
    throw std::runtime_error(
        "WebGPU scatter: output must be fp32 [1,1,262144]");
  }

  add_flat_copy(graph, input_id, output_id);
  const char* shader =
      unique_indices ? kScatterUniqueIndicesWGSL : kScatterWGSL;
  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      graph.device(),
      shader,
      {
          {0, WGPUBufferBindingType_Storage, output.buffer, output.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           index.buffer,
           index.nbytes},
          {2,
           WGPUBufferBindingType_ReadOnlyStorage,
           source.buffer,
           source.nbytes},
      });
  const uint32_t workgroups = unique_indices
      ? static_cast<uint32_t>(utils::div_up<int64_t>(
            kSelectedCount, kScatterUniqueIndicesWorkgroupSizeX))
      : 1u;
  try {
    graph.add_dispatch(
        {bundle.pipeline,
         bundle.bind_group,
         workgroups,
         unique_indices ? "scatter_unique_indices" : "scatter"});
  } catch (...) {
    wgpuComputePipelineRelease(bundle.pipeline);
    wgpuBindGroupRelease(bundle.bind_group);
    throw;
  }
}

void scatter_generic_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  scatter_impl(graph, args, false);
}

void scatter_unique_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  scatter_impl(graph, args, true);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(aten.scatter.src, scatter_generic_impl);
  WEBGPU_REGISTER_OP(et_vk.scatter_src_unique.default, scatter_unique_impl);
}

} // namespace executorch::backends::webgpu

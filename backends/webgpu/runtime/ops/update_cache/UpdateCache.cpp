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
#include <executorch/backends/webgpu/runtime/ops/update_cache/UpdateCacheState.h>
#include <executorch/backends/webgpu/runtime/ops/update_cache/update_cache_wgsl.h>

#include <webgpu/webgpu.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace executorch::backends::webgpu {

namespace {

int64_t read_input_pos(const WebGPUGraph& graph, int input_pos_id) {
  const auto type = graph.get_value_type(input_pos_id);
  if (type == WebGPUGraph::ValueType::Int) {
    return graph.get_int(input_pos_id);
  }
  if (type == WebGPUGraph::ValueType::SymInt) {
    return graph.read_symint(input_pos_id);
  }
  throw std::runtime_error(
      "WebGPU update_cache: input_pos must be Int or SymInt");
}

void validate_fp32_tensor(const WebGPUTensor& tensor, const char* label) {
  const uint64_t numel = utils::numel_of(tensor.dims);
  if (numel > std::numeric_limits<size_t>::max() / sizeof(float) ||
      tensor.nbytes != static_cast<size_t>(numel) * sizeof(float)) {
    throw std::runtime_error(
        std::string("WebGPU update_cache: ") + label + " must be fp32");
  }
}

struct UpdateCacheRefreshContext {
  int value_id;
  int cache_id;
  int input_pos_id;
  size_t expected_value_rank;
  size_t expected_cache_rank;
  uint32_t workgroup_size;
  uint32_t max_workgroups_per_dimension;
  WGPUBuffer params_buffer;
  size_t dispatch_index;
};

void refresh_update_cache(
    WebGPUGraph& graph,
    const UpdateCacheRefreshContext& context) {
  const LiveUpdateCacheInputs inputs = {
      graph.cur_dims(context.value_id),
      graph.cur_dims(context.cache_id),
      context.expected_value_rank,
      context.expected_cache_rank,
      read_input_pos(graph, context.input_pos_id),
      context.workgroup_size,
      context.max_workgroups_per_dimension,
  };
  refresh_live_update_cache_state(
      inputs, [&](const LiveUpdateCacheState& state) {
        wgpuQueueWriteBuffer(
            graph.queue(),
            context.params_buffer,
            0,
            &state.params,
            sizeof(state.params));
        auto& dispatch = graph.dispatch_at(context.dispatch_index);
        dispatch.workgroup_count_x = state.workgroup_count_x;
        dispatch.workgroup_count_y = 1;
      });
}

// llama.update_cache.default args: [value, cache, input_pos, out].
void update_cache_impl(WebGPUGraph& graph, const std::vector<int>& args) {
  const int value_id = args.at(0);
  const int cache_id = args.at(1);
  const int input_pos_id = args.at(2);

  WGPUDevice device = graph.device();

  const auto& value_tensor = graph.get_tensor(value_id);
  const auto& cache_tensor = graph.get_tensor(cache_id);
  if (value_tensor.dims.size() < 4 || cache_tensor.dims.size() < 4) {
    throw std::runtime_error("WebGPU update_cache: expects 4D value and cache");
  }
  validate_fp32_tensor(value_tensor, "value");
  validate_fp32_tensor(cache_tensor, "cache");

  // Validate dispatch against device limits before allocating GPU objects.
  const uint32_t wg_size =
      utils::clamp_workgroup_size(device, kUpdateCacheWorkgroupSizeX);
  const uint32_t max_workgroups = utils::queried_max_workgroups(device);
  const LiveUpdateCacheInputs initial_inputs = {
      value_tensor.dims,
      cache_tensor.dims,
      value_tensor.dims.size(),
      cache_tensor.dims.size(),
      read_input_pos(graph, input_pos_id),
      wg_size,
      max_workgroups,
  };
  const LiveUpdateCacheState initial_state =
      compute_live_update_cache_state(initial_inputs);
  WGPUBuffer uniform_buffer =
      graph.create_params_buffer(initial_state.params);

  WGPUConstantEntry wg_size_constant = {};
  wg_size_constant.key = {"wg_size", WGPU_STRLEN};
  wg_size_constant.value = static_cast<double>(wg_size);

  utils::ComputePipelineBundle bundle = utils::make_compute_pipeline(
      device,
      kUpdateCacheWGSL,
      {
          {0,
           WGPUBufferBindingType_Storage,
           cache_tensor.buffer,
           cache_tensor.nbytes},
          {1,
           WGPUBufferBindingType_ReadOnlyStorage,
           value_tensor.buffer,
           value_tensor.nbytes},
          {2,
           WGPUBufferBindingType_Uniform,
           uniform_buffer,
           sizeof(UpdateCacheParams)},
      },
      &wg_size_constant,
      1);

  const size_t dispatch_index = graph.add_dispatch(
      {bundle.pipeline,
       bundle.bind_group,
       initial_state.workgroup_count_x,
       "update_cache"});

  const UpdateCacheRefreshContext refresh_context = {
      value_id,
      cache_id,
      input_pos_id,
      value_tensor.dims.size(),
      cache_tensor.dims.size(),
      wg_size,
      max_workgroups,
      uniform_buffer,
      dispatch_index,
  };
  std::vector<int> symint_triggers;
  if (graph.get_value_type(input_pos_id) == WebGPUGraph::ValueType::SymInt) {
    symint_triggers.push_back(input_pos_id);
  }
  graph.add_post_resize_hook(
      {value_id, cache_id},
      symint_triggers,
      refresh_update_cache,
      refresh_context);
}

} // namespace

WEBGPU_REGISTER_OPERATORS {
  WEBGPU_REGISTER_OP(update_cache.default, update_cache_impl);
}

} // namespace executorch::backends::webgpu

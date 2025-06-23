/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/DynamicDispatchNode.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

DynamicDispatchNode::DynamicDispatchNode(
    ComputeGraph& graph,
    const PickShaderFn& pick_shader_fn,
    const PickGlobalFn& pick_global_wg_fn,
    const PickLocalFn& pick_local_wg_fn,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const std::vector<PushConstantDataInfo>& push_constants,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<ValueRef>& resize_args,
    const ResizeFunction& resize_fn)
    : DispatchNode(
          graph,
          pick_shader_fn(&graph, args, resize_args),
          {1u, 1u, 1u},
          {8u, 8u, 1u},
          args,
          params,
          push_constants,
          spec_vars,
          resize_args,
          resize_fn),
      pick_shader_fn_(pick_shader_fn),
      pick_global_wg_fn_(pick_global_wg_fn),
      pick_local_wg_fn_(pick_local_wg_fn) {
  global_workgroup_size_ =
      pick_global_wg_fn(&graph, shader_, args, resize_args);
  local_workgroup_size_ = utils::WorkgroupSize(pick_local_wg_fn(
      &graph, shader_, global_workgroup_size_, args, resize_args));
}

DynamicDispatchNode::DynamicDispatchNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const PickGlobalFn& pick_global_wg_fn,
    const PickLocalFn& pick_local_wg_fn,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const std::vector<PushConstantDataInfo>& push_constants,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<ValueRef>& resize_args,
    const ResizeFunction& resize_fn)
    : DispatchNode(
          graph,
          shader,
          pick_global_wg_fn(&graph, shader, args, resize_args),
          pick_local_wg_fn(
              &graph,
              shader,
              pick_global_wg_fn(&graph, shader, args, resize_args),
              args,
              resize_args),
          args,
          params,
          push_constants,
          spec_vars,
          resize_args,
          resize_fn),
      pick_shader_fn_{nullptr},
      pick_global_wg_fn_(pick_global_wg_fn),
      pick_local_wg_fn_(pick_local_wg_fn) {}

void DynamicDispatchNode::encode(ComputeGraph* graph) {
  if (pick_shader_fn_) {
    shader_ = pick_shader_fn_(graph, args_, resize_args_);
  }
  if (pick_global_wg_fn_) {
    global_workgroup_size_ =
        pick_global_wg_fn_(graph, shader_, args_, resize_args_);
  }
  if (pick_local_wg_fn_) {
    local_workgroup_size_ = utils::WorkgroupSize(pick_local_wg_fn_(
        graph, shader_, global_workgroup_size_, args_, resize_args_));
  }
  DispatchNode::encode(graph);
}

} // namespace vkcompute

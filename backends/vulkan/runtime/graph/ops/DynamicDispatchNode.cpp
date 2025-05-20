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
          pick_global_wg_fn(&graph, args, resize_args),
          pick_local_wg_fn(&graph, args, resize_args),
          args,
          params,
          push_constants,
          spec_vars,
          resize_args,
          resize_fn),
      pick_shader_fn_(pick_shader_fn),
      pick_global_wg_fn_(pick_global_wg_fn),
      pick_local_wg_fn_(pick_local_wg_fn) {}

void DynamicDispatchNode::encode(ComputeGraph* graph) {
  shader_ = pick_shader_fn_(graph, args_, resize_args_);
  global_workgroup_size_ = pick_global_wg_fn_(graph, args_, resize_args_);
  local_workgroup_size_ =
      utils::WorkgroupSize(pick_local_wg_fn_(graph, args_, resize_args_));
  DispatchNode::encode(graph);
}

} // namespace vkcompute

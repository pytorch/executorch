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
    const PickGwgFn& pick_gwg_fn,
    const PickLwgFn& pick_lwg_fn,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const std::vector<PushConstantDataInfo>& push_constants,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<ValueRef>& resize_args,
    const ResizeFunction& resize_fn)
    : DispatchNode(
          graph,
          pick_shader_fn(&graph, args, resize_args),
          GlobalWorkGrid({1u, 1u, 1u}, kExplicitWorkGrid),
          LocalWorkGroup(8u, 8u, 1u),
          args,
          params,
          push_constants,
          spec_vars,
          resize_args,
          resize_fn),
      pick_shader_fn_(pick_shader_fn),
      pick_gwg_fn_(pick_gwg_fn),
      pick_lwg_fn_(pick_lwg_fn) {
  gwg_ = pick_gwg_fn(&graph, shader_, args, resize_args);
  lwg_ = pick_lwg_fn(&graph, shader_, gwg_, args, resize_args);
  if (gwg_.required_lwg_size().is_valid()) {
    lwg_ = gwg_.required_lwg_size();
  }

  // Calculate dispatch grid similar to Context.cpp register_shader_dispatch
  wg_dispatch_grid_ = {
      utils::div_up(gwg_[0], lwg_[0]),
      utils::div_up(gwg_[1], lwg_[1]),
      utils::div_up(gwg_[2], lwg_[2])};
}

DynamicDispatchNode::DynamicDispatchNode(
    ComputeGraph& graph,
    const vkapi::ShaderInfo& shader,
    const PickGwgFn& pick_gwg_fn,
    const PickLwgFn& pick_lwg_fn,
    const std::vector<ArgGroup>& args,
    const vkapi::ParamsBindList& params,
    const std::vector<PushConstantDataInfo>& push_constants,
    const vkapi::SpecVarList& spec_vars,
    const std::vector<ValueRef>& resize_args,
    const ResizeFunction& resize_fn)
    : DispatchNode(
          graph,
          shader,
          GlobalWorkGrid({1u, 1u, 1u}, kExplicitWorkGrid),
          LocalWorkGroup(8u, 8u, 1u),
          args,
          params,
          push_constants,
          spec_vars,
          resize_args,
          resize_fn),
      pick_shader_fn_{nullptr},
      pick_gwg_fn_(pick_gwg_fn),
      pick_lwg_fn_(pick_lwg_fn) {
  gwg_ = pick_gwg_fn(&graph, shader_, args, resize_args);
  lwg_ = pick_lwg_fn(&graph, shader_, gwg_, args, resize_args);
  if (gwg_.required_lwg_size().is_valid()) {
    lwg_ = gwg_.required_lwg_size();
  }
  // Calculate the work group grid that will be dispatched
  wg_dispatch_grid_ = {
      utils::div_up(gwg_[0], lwg_[0]),
      utils::div_up(gwg_[1], lwg_[1]),
      utils::div_up(gwg_[2], lwg_[2])};
}

bool DynamicDispatchNode::trigger_resize(ComputeGraph* graph) {
  // DispatchNode::trigger_resize() will return true if any of the values
  // participating in this operation were updated.
  const bool any_arg_updated = DispatchNode::trigger_resize(graph);
  // Only re-compute the shader, global workgroup size, and local workgroup size
  // if any of the values participating in this operation were updated.
  // Otherwise, assume that these will not have changed.
  if (!any_arg_updated) {
    return false;
  }

  // Indicates if the shader dispatch should be changed since the last time the
  // command buffer was encoded.
  bool dispatch_params_changed = false;

  if (pick_shader_fn_) {
    vkapi::ShaderInfo new_shader = pick_shader_fn_(graph, args_, resize_args_);
    // Compare shader kernel names as a proxy for shader equality
    if (shader_.kernel_name != new_shader.kernel_name) {
      shader_ = new_shader;
      dispatch_params_changed = true;
    }
  }
  if (pick_gwg_fn_) {
    // Note that if global workgroup size changes, then the dispatch params
    // may not actually be different. The actual value to check is the
    // work group grid size that will be dispatched, which is calculated
    // below.
    gwg_ = pick_gwg_fn_(graph, shader_, args_, resize_args_);
  }
  if (pick_lwg_fn_) {
    LocalWorkGroup new_lwg =
        pick_lwg_fn_(graph, shader_, gwg_, args_, resize_args_);
    if (gwg_.required_lwg_size().is_valid()) {
      new_lwg = gwg_.required_lwg_size();
    }
    if (lwg_ != new_lwg) {
      lwg_ = new_lwg;
      dispatch_params_changed = true;
    }
  }

  // Always recompute the new dispatch grid and check if it's different
  utils::uvec3 new_wg_dispatch_grid = {
      utils::div_up(gwg_[0], lwg_[0]),
      utils::div_up(gwg_[1], lwg_[1]),
      utils::div_up(gwg_[2], lwg_[2])};

  // Check if the new dispatch grid is different from the old one
  if (wg_dispatch_grid_ != new_wg_dispatch_grid) {
    dispatch_params_changed = true;
  }
  wg_dispatch_grid_ = new_wg_dispatch_grid;

  // If any of the dispatch params have changed, then the command buffer must
  // be re-encoded.
  if (dispatch_params_changed) {
    graph->set_requires_reencode();
  }

  return true;
}

} // namespace vkcompute

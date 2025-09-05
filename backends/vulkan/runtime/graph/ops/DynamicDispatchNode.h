/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/PushConstantData.h>
#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>

namespace vkcompute {

class ComputeGraph;

/*
 * Represents a single shader execution op in a ML model.
 */
class DynamicDispatchNode final : public DispatchNode {
  friend class ComputeGraph;

 public:
  using PickShaderFn = const std::function<vkapi::ShaderInfo(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)>;
  using PickGlobalFn = const std::function<utils::uvec3(
      ComputeGraph*,
      const vkapi::ShaderInfo& shader,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)>;
  using PickLocalFn = const std::function<utils::uvec3(
      ComputeGraph*,
      const vkapi::ShaderInfo& shader,
      const utils::uvec3& global_workgroup_size,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)>;

  explicit DynamicDispatchNode(
      ComputeGraph& graph,
      const PickShaderFn& pick_shader_fn,
      const PickGlobalFn& pick_global_wg_fn,
      const PickLocalFn& pick_local_wg_fn,
      const std::vector<ArgGroup>& args,
      const vkapi::ParamsBindList& params,
      const std::vector<PushConstantDataInfo>& push_constants,
      const vkapi::SpecVarList& spec_vars,
      const std::vector<ValueRef>& resize_args,
      const ResizeFunction& resize_fn = nullptr);

  explicit DynamicDispatchNode(
      ComputeGraph& graph,
      const vkapi::ShaderInfo& shader,
      const PickGlobalFn& pick_global_wg_fn,
      const PickLocalFn& pick_local_wg_fn,
      const std::vector<ArgGroup>& args,
      const vkapi::ParamsBindList& params,
      const std::vector<PushConstantDataInfo>& push_constants,
      const vkapi::SpecVarList& spec_vars,
      const std::vector<ValueRef>& resize_args,
      const ResizeFunction& resize_fn = nullptr);

  ~DynamicDispatchNode() override = default;

  bool trigger_resize(ComputeGraph* graph) override;

 protected:
  const PickShaderFn pick_shader_fn_;
  const PickGlobalFn pick_global_wg_fn_;
  const PickLocalFn pick_local_wg_fn_;

  utils::uvec3 wg_dispatch_grid_{1u, 1u, 1u};

 public:
  operator bool() const {
    return shader_;
  }
};

} // namespace vkcompute

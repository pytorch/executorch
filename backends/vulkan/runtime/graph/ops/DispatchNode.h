/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

class ComputeGraph;

/*
 * Represents a single shader execution op in a ML model.
 */
class DispatchNode final : public ExecuteNode {
  friend class ComputeGraph;

 public:
  explicit DispatchNode(
      ComputeGraph& graph,
      const vkapi::ShaderInfo& shader,
      const utils::uvec3& global_workgroup_size,
      const utils::uvec3& local_workgroup_size,
      const std::vector<ArgGroup>& args,
      const vkapi::ParamsBindList& params,
      const vkapi::SpecVarList& spec_vars = {},
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {});

  ~DispatchNode() override = default;

  void encode(ComputeGraph* graph) override;

 protected:
  const vkapi::ShaderInfo shader_;
  const utils::uvec3 global_workgroup_size_;
  const utils::uvec3 local_workgroup_size_;
  const vkapi::ParamsBindList params_;
  const vkapi::SpecVarList spec_vars_;

 public:
  operator bool() const {
    return shader_;
  }
};

} // namespace vkcompute

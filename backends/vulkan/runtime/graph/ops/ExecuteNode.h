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

namespace vkcompute {

class ComputeGraph;

/*
 * Represents a group of shader arguments (images and/or buffers), with a common
 * access permission.
 */
struct ArgGroup {
  ArgGroup(const ValueRef ref, const vkapi::MemoryAccessType access)
      : refs{ref}, access(access) {}

  ArgGroup(
      const std::vector<ValueRef>& refs,
      const vkapi::MemoryAccessType access)
      : refs(refs), access(access) {}

  const std::vector<ValueRef> refs;
  const vkapi::MemoryAccessType access;
};

/*
 * Represents a single execution op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of the shader corresponding to the op into the command buffer of a
 * ComputeGraph.
 */
class ExecuteNode final {
  friend class ComputeGraph;

 public:
  using ResizeFunction = const std::function<void(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)>;

  explicit ExecuteNode(
      ComputeGraph& graph,
      const vkapi::ShaderInfo& shader,
      const utils::uvec3& global_workgroup_size,
      const utils::uvec3& local_workgroup_size,
      const std::vector<ArgGroup>& args,
      const vkapi::ParamsBindList& params,
      const vkapi::SpecVarList& spec_vars = {},
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {});

  /*
   * This overload of the ExecuteNode constructor is used to register ops which
   * update a tensor view. No shader is dispatched, but the node still needs to
   * update the view's sizes and strides after a resize.
   */
  explicit ExecuteNode(
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {});

  ~ExecuteNode() = default;

  void encode(ComputeGraph* graph);

  inline void trigger_resize(ComputeGraph* graph) {
    if (resize_fn_ != nullptr) {
      resize_fn_(graph, args_, resize_args_);
    }
  }

  inline void set_node_id(uint32_t node_id) {
    node_id_ = node_id;
  }

 protected:
  uint32_t node_id_;
  const vkapi::ShaderInfo shader_;
  const utils::uvec3 global_workgroup_size_;
  const utils::uvec3 local_workgroup_size_;
  const std::vector<ArgGroup> args_;
  const vkapi::ParamsBindList params_;
  const vkapi::SpecVarList spec_vars_;
  const ResizeFunction resize_fn_;
  const std::vector<ValueRef> resize_args_;

 public:
  operator bool() const {
    return shader_;
  }
};

} // namespace vkcompute

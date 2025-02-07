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
  ArgGroup(const ValueRef ref, const vkapi::MemoryAccessFlags access)
      : refs{ref}, access(access) {}

  ArgGroup(
      const std::vector<ValueRef>& refs,
      const vkapi::MemoryAccessFlags access)
      : refs(refs), access(access) {}

  const std::vector<ValueRef> refs;
  const vkapi::MemoryAccessFlags access;
};

/*
 * Represents a single execution op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of the shader corresponding to the op into the command buffer of a
 * ComputeGraph.
 */
class ExecuteNode {
  friend class ComputeGraph;

 public:
  using ResizeFunction = const std::function<void(
      ComputeGraph*,
      const std::vector<ArgGroup>&,
      const std::vector<ValueRef>&)>;

  /*
   * This overload of the DispatchNode constructor is used to register ops which
   * update a tensor view. No shader is dispatched, but the node still needs to
   * update the view's sizes and strides after a resize.
   */
  explicit ExecuteNode(
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {},
      const std::vector<ArgGroup>& args = {},
      const std::string& name = "Graph Node");

  virtual ~ExecuteNode() = default;

  virtual void encode(ComputeGraph* graph) {
    (void)graph;
  }

  inline void trigger_resize(ComputeGraph* graph) {
    if (resize_fn_ != nullptr) {
      resize_fn_(graph, args_, resize_args_);
    }
  }

  inline void set_node_id(uint32_t node_id) {
    node_id_ = node_id;
  }

  inline const std::string& name() const {
    return name_;
  }

 protected:
  uint32_t node_id_;
  const ResizeFunction resize_fn_;
  const std::vector<ValueRef> resize_args_;
  const std::vector<ArgGroup> args_;
  const std::string name_;
};

} // namespace vkcompute

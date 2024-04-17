//123123

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

#include <optional>

namespace vkcompute {

class ComputeGraph;

/*
 * Represents a group of shader arguments (images and/or buffers), with a common
 * access permission.
 */
struct ArgGroup {
  ArgGroup(const ValueRef ref, const api::MemoryAccessType access)
      : refs{ref}, access(access) {}

  ArgGroup(const ArgGroup& ag): refs(ag.refs), access(ag.access) {}

  ArgGroup(
      const std::vector<ValueRef>& refs,
      const api::MemoryAccessType access)
      : refs(refs), access(access) {}

  const std::vector<ValueRef> refs;
  const api::MemoryAccessType access;
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

  ExecuteNode(
      ComputeGraph& graph,
      const api::ShaderInfo& shader,
      const api::utils::uvec3& global_workgroup_size,
      const api::utils::uvec3& local_workgroup_size,
      const std::vector<ArgGroup>& args,
      const std::vector<std::shared_ptr<api::UniformParamsBuffer>>& params,
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {});

  ExecuteNode(
      ComputeGraph& graph,
      const ArgGroup& src,
      const ArgGroup& dst,
      const api::utils::uvec3& copy_range,
      const api::utils::uvec3& src_offset,
      const api::utils::uvec3& dst_offset);

  ~ExecuteNode() = default;

  void encode(ComputeGraph* graph);

  inline void trigger_resize(ComputeGraph* graph) {
    if (resize_fn_ != nullptr) {
      resize_fn_(graph, args_, resize_args_);
    }
  }

 protected:
  const api::ShaderInfo shader_;
  const std::optional<api::utils::uvec3> global_workgroup_size_;
  const std::optional<api::utils::uvec3> local_workgroup_size_;
  const std::vector<ArgGroup> args_;
  std::vector<std::shared_ptr<api::UniformParamsBuffer>> params_;
  const ResizeFunction resize_fn_;
  const std::vector<ValueRef> resize_args_;

  const std::optional<ArgGroup> src_;
  const std::optional<ArgGroup> dst_;
  const std::optional<api::utils::uvec3> copy_range_;
  const std::optional<api::utils::uvec3> src_offset_;
  const std::optional<api::utils::uvec3> dst_offset_;

 private:
  void encode_shader(ComputeGraph *graph);
  void encode_copy(ComputeGraph *graph);
};

} // namespace vkcompute

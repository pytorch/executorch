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
 * Represents a single prepacking op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of shaders transferring necessary data (such as weights and biases)
 * to the GPU.
 */
class PrepackNode final {
  friend class ComputeGraph;

 public:
  PrepackNode(
      ComputeGraph& graph,
      const api::ShaderInfo& shader,
      const api::utils::uvec3& global_workgroup_size,
      const api::utils::uvec3& local_workgroup_size,
      const ValueRef tref,
      const ValueRef packed,
      const std::vector<std::shared_ptr<api::UniformParamsBuffer>>& params);

  ~PrepackNode() = default;

  void encode(ComputeGraph* graph);

 protected:
  const api::ShaderInfo shader_;
  api::ShaderInfo noop_shader_;
  const api::utils::uvec3 global_workgroup_size_;
  const api::utils::uvec3 local_workgroup_size_;
  const ValueRef tref_;
  const ValueRef packed_;
  // TODO(T180906457): allow re-computing param buffers.
  std::vector<std::shared_ptr<api::UniformParamsBuffer>> params_;
};

} // namespace vkcompute

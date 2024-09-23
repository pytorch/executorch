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
      const vkapi::ShaderInfo& shader,
      const utils::uvec3& global_workgroup_size,
      const utils::uvec3& local_workgroup_size,
      const ValueRef tref,
      const ValueRef packed,
      const vkapi::ParamsBindList& params,
      const vkapi::SpecVarList& spec_vars = {});

  ~PrepackNode() = default;

  void encode(ComputeGraph* graph);

  inline void set_node_id(uint32_t node_id) {
    node_id_ = node_id;
  }

 protected:
  uint32_t node_id_;
  const vkapi::ShaderInfo shader_;
  vkapi::ShaderInfo noop_shader_;
  const utils::uvec3 global_workgroup_size_;
  const utils::uvec3 local_workgroup_size_;
  const ValueRef tref_;
  const ValueRef packed_;
  const vkapi::ParamsBindList params_;
  const vkapi::SpecVarList spec_vars_;

 private:
  api::StagingBuffer create_staging_buffer(ComputeGraph* graph);
};

} // namespace vkcompute

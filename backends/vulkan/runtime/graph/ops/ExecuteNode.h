/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Tensor.h>
#include <ATen/native/vulkan/api/Types.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

namespace at {
namespace native {
namespace vulkan {

class ComputeGraph;

/*
 * Represents a single execution op in a ML model. In graph mode, ops will be
 * implemented in a derived class that implements encode, which will implement
 * encoding of the shader corresponding to the op into the command buffer of a
 * ComputeGraph.
 */
class ExecuteNode final {
  friend class ComputeGraph;

 public:
  ExecuteNode(
      const api::ShaderInfo& shader,
      const api::utils::uvec3& global_workgroup_size,
      const api::utils::uvec3& local_workgroup_size,
      const std::vector<ValueRef>& outputs,
      const std::vector<ValueRef>& inputs,
      api::UniformParamsBuffer&& params)
      : shader_(shader),
        global_workgroup_size_(global_workgroup_size),
        local_workgroup_size_(local_workgroup_size),
        outputs_(outputs),
        inputs_(inputs),
        params_(std::move(params)) {}

  ~ExecuteNode() = default;

  void encode(ComputeGraph* graph);

 protected:
  const api::ShaderInfo shader_;
  const api::utils::uvec3 global_workgroup_size_;
  const api::utils::uvec3 local_workgroup_size_;
  const std::vector<ValueRef> outputs_;
  const std::vector<ValueRef> inputs_;
  // TODO(T180906086): pass multiple buffers and index with ValueRef.
  // TODO(T180906457): allow re-computing param buffers.
  api::UniformParamsBuffer params_;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

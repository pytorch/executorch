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

#include <executorch/backends/vulkan/runtime/graph/Value.h>

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
class ExecuteNode {
  friend class ComputeGraph;

 public:
  ExecuteNode(ValueRef input, ValueRef output)
      : inputs_{input}, outputs_{output} {}
  ExecuteNode(
      const std::vector<ValueRef>& inputs,
      const std::vector<ValueRef>& outputs)
      : inputs_(inputs), outputs_(outputs) {}

  virtual ~ExecuteNode() = default;

 protected:
  std::vector<ValueRef> inputs_;
  std::vector<ValueRef> outputs_;

 public:
  virtual void encode(ComputeGraph* graph) const = 0;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

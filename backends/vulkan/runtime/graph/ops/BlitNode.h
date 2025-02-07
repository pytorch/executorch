/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/containers/Value.h>

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

/*
 * Represents a tensor blit execution op in a ML model.
 */
class BlitNode final : public ExecuteNode {
  friend class ComputeGraph;

 public:
  explicit BlitNode(
      ComputeGraph& graph,
      ValueRef src,
      ValueRef dst,
      /*const vkapi::ScalarType& dtype,*/
      const ResizeFunction& resize_fn = nullptr,
      const std::vector<ValueRef>& resize_args = {});

  ~BlitNode() override = default;

  void encode(ComputeGraph* graph) override;

 protected:
  ValueRef src_;
  ValueRef dst_;
  // const vkapi::ScalarType &dtype_;
};

} // namespace vkcompute

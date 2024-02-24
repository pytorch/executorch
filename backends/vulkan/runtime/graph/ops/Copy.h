/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_VULKAN_API

#include <executorch/backends/vulkan/runtime/graph/Graph.h>

namespace at {
namespace native {
namespace vulkan {

void add_copy_node(ComputeGraph& graph, const ValueRef from, const ValueRef to);
ValueRef add_copy_node(ComputeGraph& graph, const ValueRef from);

class CopyNode : public virtual OpNode {
 public:
  explicit CopyNode(const ValueRef from, const ValueRef to);

  void encode_execute(ComputeGraph* graph) const override;
};

} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */

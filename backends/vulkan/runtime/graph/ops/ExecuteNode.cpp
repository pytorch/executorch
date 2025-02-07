/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {
ExecuteNode::ExecuteNode(
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args,
    const std::vector<ArgGroup>& args,
    const std::string& name)
    : resize_fn_(resize_fn),
      resize_args_(resize_args),
      args_(args),
      name_(name) {}
} // namespace vkcompute

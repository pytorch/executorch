/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

namespace vkcompute {

bool OperatorRegistry::has_op(const std::string& name) {
  return table_.count(name) > 0;
}

OperatorRegistry::OpFunction& OperatorRegistry::get_op_fn(
    const std::string& name) {
  const auto it = table_.find(name);
  VK_CHECK_COND(it != table_.end(), "Could not find operator with name ", name);
  return it->second;
}

void OperatorRegistry::register_op(const std::string& name, OpFunction& fn) {
  table_.insert(std::make_pair(name, fn));
}

OperatorRegistry& operator_registry() {
  static OperatorRegistry registry;
  return registry;
}

} // namespace vkcompute

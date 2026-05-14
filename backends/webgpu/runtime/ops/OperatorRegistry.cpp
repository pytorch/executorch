/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/ops/OperatorRegistry.h>

#include <stdexcept>

namespace executorch {
namespace backends {
namespace webgpu {

bool OperatorRegistry::has_op(const std::string& name) {
  return table_.count(name) > 0;
}

OpFunction& OperatorRegistry::get_op_fn(const std::string& name) {
  const auto it = table_.find(name);
  if (it == table_.end()) {
    throw std::runtime_error(
        "WebGPU OperatorRegistry: could not find operator: " + name);
  }
  return it->second;
}

void OperatorRegistry::register_op(
    const std::string& name,
    const OpFunction& fn) {
  table_.insert(std::make_pair(name, fn));
}

OperatorRegistry& webgpu_operator_registry() {
  static OperatorRegistry registry;
  return registry;
}

} // namespace webgpu
} // namespace backends
} // namespace executorch

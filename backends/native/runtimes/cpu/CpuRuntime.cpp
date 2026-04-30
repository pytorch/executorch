/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/runtimes/cpu/CpuRuntime.h>

#include <executorch/backends/native/runtimes/cpu/CpuEngine.h>
#include <executorch/backends/native/runtimes/cpu/CpuOpRegistry.h>

#include <string>

namespace executorch {
namespace backends {
namespace native {

bool CpuRuntime::can_run(const OpDescriptor& op) const {
  std::string name(op.name);
  // Allowlist filter (test/dev mode).
  if (!supported_ops_.empty() && supported_ops_.count(name) == 0) {
    return false;
  }
  // Default: accept any op the portable kernel registry knows about.
  return ::executorch::backends::portable::cpu_op_registry().has_op(name);
}

std::unique_ptr<Engine> CpuRuntime::instantiate() {
  InstanceId id = next_instance_id_.fetch_add(1, std::memory_order_relaxed);
  return std::make_unique<CpuEngine>(ctx_, id);
}

} // namespace native
} // namespace backends
} // namespace executorch

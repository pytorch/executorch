/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Instance.h>
#include <executorch/backends/portable/runtime_v2/api/Location.h>
#include <executorch/backends/portable/runtime_v2/api/OpDescriptor.h>
#include <executorch/backends/portable/runtime_v2/api/RuntimeContext.h>

#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>

#include <memory>
#include <string_view>

namespace executorch {
namespace backends {
namespace portable_v2 {

class ProviderRegistry;

/**
 * Process-wide singleton per runtime (CPU, Metal, Vulkan).
 * Queryable for capabilities; factory for Instances and Buffers.
 *
 * See §4.5 of the design doc.
 */
class Provider {
 public:
  virtual ~Provider() = default;
  virtual std::string_view name() const = 0;

  // Assigned by the ProviderRegistry on registration; unique per process.
  // Never to be hardcoded. This is the opaque identity (used in
  // Location); the dense per-Plan RuntimeIndex is assigned at route()
  // time.
  RuntimeId id() const { return id_; }

  virtual bool is_available_on_device() const = 0;

  // Capability query. Cheap; called O(num_ops × num_providers) at routing.
  // The Provider sees the OpDescriptor and decides accept/reject.
  // Returns true iff this Provider can execute the op. Cost ranking can
  // be added later by replacing bool with optional<OpCapability>; both
  // OpDescriptor and the return type are designed to grow non-breakingly.
  virtual bool can_run(const OpDescriptor& op) const = 0;

  // Process-wide state (pools, kernel caches, command stream).
  // Lazy-initialized on first instantiate(); lives until process exit.
  virtual RuntimeContext& context() = 0;

  // Per-program factory. Holds non-owning RuntimeContext& from this
  // Provider.
  virtual std::unique_ptr<Instance> instantiate() = 0;

 private:
  friend class ProviderRegistry;
  RuntimeId id_ = 0;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

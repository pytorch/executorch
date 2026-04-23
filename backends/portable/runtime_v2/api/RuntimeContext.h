/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Process-wide state owned by a Provider (pools, kernel caches, GPU
 * queues, command streams, per-Instance submission tracker).
 *
 * Tag base class; each concrete runtime subclasses with its own state.
 * Survives across multiple loaded programs.
 *
 * See §4.8 of the design doc.
 */
class RuntimeContext {
 public:
  virtual ~RuntimeContext() = default;
};

// Unique within a single RuntimeContext (NOT process-wide). Used by
// SubmissionTracker to scope drain() to a single Instance's submissions.
using InstanceId = uint32_t;

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

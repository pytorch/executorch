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
namespace native {

/**
 * Process-wide state owned by a Runtime (pools, kernel caches, GPU
 * queues, command streams, per-Engine submission tracker).
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
// SubmissionTracker to scope drain() to a single Engine's submissions.
using InstanceId = uint32_t;

} // namespace native
} // namespace backends
} // namespace executorch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Router.h>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Default greedy router. See §4.10 of PORTABLE_BACKEND_API_PROPOSAL.md.
 *
 * Algorithm:
 *   1. Build dense index space (CPU at index 0; others by registration order).
 *   2. For each instruction, ask each Provider in priority order; pick the
 *      first whose can_run() returns true.
 *   3. Group consecutive same-runtime instructions into segments; call
 *      Instance::compile_segment for each.
 *   4. For each value crossing a segment boundary: synthesize a destination
 *      value_id on the consumer side, emit a TransferStep, and emit an
 *      AllocRequest carrying the source's host buffer as a host_alias hint
 *      (the executor patches the actual Buffer* before allocate_all).
 *   5. Upload constants via Instance::upload_constant.
 *   6. Emit per-provider AllocRequest lists into Plan::alloc_plans (the
 *      executor's allocate_buffers step performs the actual allocation
 *      host-first so device requests can resolve their host_alias hints).
 *   7. Build Plan::inputs / outputs.
 *
 * v1 scope:
 *   - CPU-only happy path: one segment, no TransferSteps, host-aliased
 *     inputs/outputs (host-addressable runtimes alias caller storage zero-copy), constants
 *     uploaded via NDM (zero-copy alias).
 *   - Multi-provider routing: skeleton in place but the cross-runtime
 *     transfer machinery is left as TODO until the first non-CPU
 *     Provider lands.
 */
class GreedyRouter final : public Router {
 public:
  ::executorch::runtime::Result<Plan> route(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<Provider* const> providers,
      ::executorch::runtime::Span<Instance* const> instances,
      const ::executorch::runtime::NamedDataMap* ndm,
      const RouterOptions& options) override;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

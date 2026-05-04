/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Router.h>

namespace executorch {
namespace backends {
namespace native {

/**
 * Default greedy router. See §4.10 of PORTABLE_BACKEND_API_PROPOSAL.md.
 *
 * Algorithm:
 *   1. Build dense index space (CPU at index 0; others by registration order).
 *   2. For each instruction, ask each Runtime in priority order; pick the
 *      first whose can_run() returns true.
 *   3. Group consecutive same-runtime instructions into segments; call
 *      Engine::compile_segment for each.
 *   4. For each value crossing a segment boundary: mint a destination
 *      value_id on the consumer side, emit a TransferStep, and emit an
 *      AllocRequest carrying the source's host buffer as a mirror_partner
 *      hint (the executor patches the actual Buffer* before
 *      allocate_buffers).
 *   5. Emit per-provider const_plans (ConstRequest lists) into Plan.
 *      Constant uploads themselves are driven post-route by
 *      NativeBackend::upload_constants — route() stays pure planning.
 *   6. Emit per-provider AllocRequest lists into Plan::alloc_plans (the
 *      executor's allocate_buffers step performs the actual allocation
 *      host-first so device requests can resolve their mirror_partner hints).
 *   7. Build Plan::inputs / outputs.
 *
 * v1 scope:
 *   - CPU-only happy path: one segment, no TransferSteps, host-aliased
 *     inputs/outputs (host-addressable runtimes alias caller storage
 * zero-copy), constants uploaded via NDM (zero-copy alias).
 *   - Multi-provider routing: skeleton in place but the cross-runtime
 *     transfer machinery is left as TODO until the first non-CPU
 *     Runtime lands.
 */
class GreedyRouter final : public Router {
 public:
  ::executorch::runtime::Result<Plan> route(
      const ::executorch::backends::portable::Graph& graph,
      ::executorch::runtime::Span<Runtime* const> providers,
      ::executorch::runtime::Span<Engine* const> instances,
      const ::executorch::runtime::NamedDataMap* ndm,
      const RouterOptions& options) override;
};

} // namespace native
} // namespace backends
} // namespace executorch

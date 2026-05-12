/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * GreedyRouter — slim orchestrator. Each phase lives in its own
 * sibling .cpp file (see GreedyRouterContext.h for the full list).
 * route() allocates a RouterContext and walks the phases in order.
 */

#include <executorch/backends/native/routers/GreedyRouter.h>

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace native {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

Result<Plan> GreedyRouter::route(
    const ::executorch::backends::portable::Graph& graph,
    Span<Runtime* const> providers,
    Span<Engine* const> instances,
    const ::executorch::runtime::NamedDataMap* ndm,
    const RouterOptions& options) {
  if (providers.size() != instances.size())
    return Error::InvalidArgument;
  if (providers.empty())
    return Error::InvalidArgument;

  Plan plan;
  plan.providers.assign(providers.begin(), providers.end());
  plan.instances.assign(instances.begin(), instances.end());
  plan.max_hops = options.max_hops;

  router_internal::RouterContext ctx{
      graph, providers, instances, ndm, options, plan};

  if (auto e = router_internal::assign_and_segment(ctx); e != Error::Ok)
    return e;
  router_internal::plan_homes_and_mirrors(ctx);
  if (auto e = router_internal::emit_allocs(ctx); e != Error::Ok)
    return e;
  router_internal::plan_transfers(ctx);
  if (auto e = router_internal::compile_and_emit_steps(ctx); e != Error::Ok)
    return e;
  if (!ctx.control_instrs.empty()) {
    if (auto e = router_internal::interleave_control_flow(ctx); e != Error::Ok)
      return e;
  }

  // Terminal events = last signal on each Engine.
  for (const auto& kv : ctx.last_signal_per_inst) {
    plan.terminal_events.push_back(kv.second);
  }

  if (options.dump_trace) {
    ET_LOG(
        Info,
        "GreedyRouter: %zu segments / %zu steps",
        ctx.segments.size(),
        plan.steps.size());
  }

  return plan;
}

} // namespace native
} // namespace backends
} // namespace executorch

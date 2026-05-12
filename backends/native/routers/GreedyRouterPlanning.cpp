/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/runtime/platform/log.h>

#include <string>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {
using ::executorch::backends::portable::ValueType;
} // namespace

void plan_homes_and_mirrors(RouterContext& ctx) {
  const auto& graph = ctx.graph;

  // ---- 5. HomePlanner: decide where each value's "home" Buffer lives ----
  // Three rules, applied in order:
  //   - Constant                → not in value_home_provider; uploaded
  //                               per consuming runtime via const_plans.
  //   - Graph IO                → home = host (slot 0). Caller's data_ptr
  //                               binds via bind_inputs/bind_outputs on
  //                               the host pool, so a host-side canonical
  //                               Buffer must exist regardless of which
  //                               runtimes touch the value.
  //   - Otherwise               → home = the unique runtime that touches
  //                               the value (producer + consumers), or
  //                               host if multiple runtimes touch it.
  //
  // The third rule unifies what used to be separate "intermediate" and
  // "mutable buffer placeholder" branches: a placeholder is just a value
  // with no producer in the delegate, and the touching-runtimes union
  // computes the right home for both shapes.
  //
  // Cross-runtime values landing on host means every device-side mirror
  // sources from one place — host-first allocation order naturally
  // satisfies dependencies, and no peer-to-peer transfer is ever needed.
  //
  // NOTE: there is no special "aliased mem_obj_id" home rule. We rely
  // on the AOT reinplace_pass to fold buffer mutations into in-place
  // ops (`aten::index_put_`, etc.), which collapses the historical
  // (placeholder, mutation_source) two-vid alias group into a single
  // vid.
  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor)
      continue;
    if (graph.is_constant(i))
      continue; // constant
    if (ctx.io_ids.count(i) > 0) {
      ctx.value_home_provider[i] = 0; // graph IO → host
      continue;
    }

    // Touching = {producer's runtime if exists} ∪ {consumer runtimes}.
    std::set<int> touching;
    auto pit = ctx.value_producer_seg.find(i);
    if (pit != ctx.value_producer_seg.end()) {
      touching.insert(ctx.segments[pit->second].provider_idx);
    }
    auto cit = ctx.value_consumer_providers.find(i);
    if (cit != ctx.value_consumer_providers.end()) {
      for (int c : cit->second)
        touching.insert(c);
    }

    if (touching.empty())
      continue; // truly unused
    ctx.value_home_provider[i] = (touching.size() == 1) ? *touching.begin() : 0;
  }

  // ---- 6. MirrorPlanner ------------------------------------------------
  //
  // Pre-compute the full set of (value_id, runtime) → mirror_id pairs.
  // A mirror is needed whenever a segment touches a value whose home
  // runtime differs from the segment's runtime. Constants are excluded
  // (they live in const_plans, materialized once per consuming runtime
  // via upload_constants).
  //
  // This rule covers:
  //   - Cross-runtime intermediates (home = host; non-host segment
  //     reads/writes them).
  //   - Graph IO (home = host; non-host segment reads inputs / writes
  //     outputs).
  //
  // All mirrors are paired with a host-homed source vid, so the only
  // cross-runtime transfer primitives needed are upload_from_host /
  // download_to_host. There are no peer-to-peer mirror pairs.
  //
  // Output: mirror_table (lookup) and a parallel mirror_values list
  // (recorded into plan.mirror_values during alloc emission below).
  // Both AllocPlanner (Phase 7) and TransferPlanner (Phase 8) read
  // mirror_table; neither mints new entries.
  ctx.next_mirror_id = static_cast<uint32_t>(graph.num_values());

  auto needs_mirror_on = [&](uint32_t v, int p) -> bool {
    if (graph.value_type(v) != ValueType::Tensor)
      return false;
    if (graph.is_constant(v))
      return false;
    auto hit = ctx.value_home_provider.find(v);
    if (hit == ctx.value_home_provider.end())
      return false;
    if (hit->second == p)
      return false;

    // For graph IO, defer to the engine: it may be able to consume the
    // host pool's IO Buffer directly (e.g., CPU; UMA Metal with
    // newBufferWithBytesNoCopy) and skip the mirror entirely. Default
    // is false (router mints a mirror), so engines that haven't
    // overridden behave conservatively.
    if (ctx.graph_input_ids.count(v)) {
      return !ctx.instances[p]->handles_input_directly(v);
    }
    if (ctx.graph_output_ids.count(v)) {
      return !ctx.instances[p]->handles_output_directly(v);
    }
    // Non-IO cross-runtime intermediate: always mirror.
    return true;
  };

  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    int cur = ctx.segments[s].provider_idx;
    auto handle = [&](uint32_t v) {
      if (!needs_mirror_on(v, cur))
        return;
      auto key = std::make_pair(v, cur);
      if (ctx.mirror_table.count(key))
        return; // already minted for this (v, cur)
      uint32_t mid = ctx.next_mirror_id++;
      ctx.mirror_table[key] = mid;
      ET_LOG(
          Debug,
          "[mem] router: minted mirror seg=%zu value_id=%u -> mirror_id=%u (%s)",
          s,
          v,
          mid,
          std::string(ctx.providers[cur]->name()).c_str());
    };
    for (uint32_t v : ctx.segments[s].input_value_ids)
      handle(v);
    for (uint32_t v : ctx.segments[s].output_value_ids)
      handle(v);
  }
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch

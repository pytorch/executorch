/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouterContext.h>

#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {
using ::executorch::backends::portable::ValueType;
using ::executorch::runtime::Error;
} // namespace

Error emit_allocs(RouterContext& ctx) {
  const auto& graph = ctx.graph;
  auto& plan = ctx.plan;

  // ---- 7. Emit const-plans + alloc-plans for intermediates ------------
  // Constants are partitioned per provider (one ConstRequest per
  // (provider, value_id) for every provider that consumes the value).
  // The router emits these as planning output into plan.const_plans;
  // NativeBackend's post-route upload_constants pass drives the actual
  // engine calls (symmetric to alloc_plans / materialize_buffers).
  // Intermediates are NOT allocated here — the router emits an
  // AllocRequest entry into plan.alloc_plans[home_provider]; the
  // executor's allocate_buffers step (called after route + materialize)
  // performs the actual allocation via allocate_buffers.

  // Initialize alloc_plans and const_plans (one entry per provider).
  plan.alloc_plans.assign(ctx.providers.size(), {});
  plan.const_plans.assign(ctx.providers.size(), {});

  // (provider_idx, mem_obj_id) -> sentinel to track if mem_obj_id already
  // emitted for this provider (so we emit at most one AllocRequest per
  // (provider, mem_obj_id) group).
  std::set<std::pair<int, int32_t>> mem_id_emitted;

  // ---- 6a. Constants: bucket per consuming provider into plan.const_plans.
  // Each (provider, value_id) consumed by that provider gets one
  // ConstRequest. NativeBackend will drive the upload_constants calls
  // post-route. This keeps route() pure: no engine I/O during planning.
  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor)
      continue;

    if (const char* key = graph.tensor_constant_data_key(i); key != nullptr) {
      // NDM-stored constant. Schedule upload on every consuming provider.
      if (!ctx.ndm) {
        ET_LOG(Error, "GreedyRouter: constant '%s' needs NDM", key);
        return Error::InvalidArgument;
      }
      auto it = ctx.value_consumer_providers.find(i);
      if (it == ctx.value_consumer_providers.end()) {
        // Unused constant — no consumer. Don't emit a ConstRequest;
        // any provider would just waste a Buffer wrapper on bytes
        // nothing reads.
        continue;
      }
      for (int p : it->second) {
        plan.const_plans[p].push_back(Engine::ConstRequest{
            /*value_id=*/i,
            /*ndm_key=*/key,
            /*inline_data=*/{}});
        ET_LOG(
            Debug,
            "[mem] router: const-plan value_id=%u key='%s' provider=%d (%s)",
            i,
            key,
            p,
            std::string(ctx.providers[p]->name()).c_str());
      }
      continue;
    }

    // Inline constant: bytes live in program.constant_buffer (not
    // promoted to NDM by AOT). Schedule upload with the inline data
    // span; the engine aliases it without an NDM lookup.
    if (auto inline_bytes = graph.tensor_inline_data(i);
        !inline_bytes.empty()) {
      auto it = ctx.value_consumer_providers.find(i);
      if (it == ctx.value_consumer_providers.end())
        continue;
      for (int p : it->second) {
        plan.const_plans[p].push_back(Engine::ConstRequest{
            /*value_id=*/i, /*ndm_key=*/{}, /*inline_data=*/inline_bytes});
        ET_LOG(
            Debug,
            "[mem] router: const-plan value_id=%u inline bytes=%zu provider=%d (%s)",
            i,
            inline_bytes.size(),
            p,
            std::string(ctx.providers[p]->name()).c_str());
      }
      continue;
    }

    if (ctx.io_ids.count(i) > 0)
      continue; // emitted in graph-IO loop below

    // Intermediate: emit AllocRequest on the value's home provider.
    auto hit = ctx.value_home_provider.find(i);
    if (hit == ctx.value_home_provider.end())
      continue; // unused
    int home_p = hit->second;

    int32_t mem_id = graph.mem_obj_id(i);
    size_t nbytes = graph.tensor_nbytes_max(i);
    if (nbytes == 0)
      continue;

    // Always emit an AllocRequest per value_id. Even if mem_id is
    // shared with another value (e.g., AOT memory planner aliased them
    // because their lifetimes don't overlap, or because of an op like
    // aten::copy_'s formal `out` sharing data with `src`), each value_id
    // gets its own binding entry. Backends that want to honor mem_id
    // sharing as actual storage aliasing (Vulkan SharedObject style)
    // can do so internally based on req.mem_obj_id; for our current
    // backends each value_id gets its own Buffer.
    Engine::AllocRequest req;
    req.value_id = i;
    req.mem_obj_id = mem_id;
    // Intermediates with home == host always have at least one non-host
    // touching runtime (HostPool can't compute), so they are always
    // HostMirror. Intermediates with home == device runtime are
    // DeviceOnly.
    req.kind = (home_p == 0) ? MemoryKind::HostMirror : MemoryKind::DeviceOnly;
    plan.alloc_plans[home_p].push_back(req);
    ET_LOG(
        Debug,
        "[mem] router: alloc-request intermediate value_id=%u home_provider=%d (%s) kind=%s mem_id=%d nbytes=%zu",
        i,
        home_p,
        std::string(ctx.providers[home_p]->name()).c_str(),
        to_string(req.kind),
        mem_id,
        nbytes);
  }

  // (Constants are NOT uploaded here — see plan.const_plans, driven by
  // NativeBackend post-route.)

  // ---- 7b. Emit DeviceMirror AllocRequests from the pre-built
  //          mirror_table. Every (vid, runtime) entry becomes one
  //          DeviceMirror alloc on that runtime, paired back to the
  //          host-homed source via host_mirror_value_id.
  //
  //          Iteration is sorted by mirror_id (insertion order) so
  //          plan.mirror_values is populated deterministically.
  {
    std::vector<std::pair<std::pair<uint32_t, int>, uint32_t>> entries(
        ctx.mirror_table.begin(), ctx.mirror_table.end());
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });
    for (const auto& kv : entries) {
      uint32_t v = kv.first.first;
      int dst_p = kv.first.second;
      uint32_t v_mirror = kv.second;
      Engine::AllocRequest req;
      req.value_id = v_mirror;
      req.mem_obj_id = graph.mem_obj_id(v);
      req.host_mirror_value_id = v;
      req.kind = MemoryKind::DeviceMirror;
      plan.alloc_plans[dst_p].push_back(req);
    }
  }

  // Add IO destination requests on the host provider (slot 0). Graph IO
  // is always HostExtern (caller-owned per-execute storage; HostPool
  // wraps via Aliasing HostBuffer). mem_obj_id is preserved for
  // diagnostics; HostPool ignores it for HostExtern.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    Engine::AllocRequest req;
    req.value_id = graph.input_id(i);
    req.mem_obj_id = graph.mem_obj_id(req.value_id);
    req.kind = MemoryKind::HostExtern;
    plan.alloc_plans[0].push_back(req);
    ET_LOG(
        Debug,
        "[mem] router: alloc-request graph input value_id=%u provider=0 (host) kind=%s mem_id=%d",
        req.value_id,
        to_string(req.kind),
        req.mem_obj_id);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    Engine::AllocRequest req;
    req.value_id = graph.output_id(i);
    req.mem_obj_id = graph.mem_obj_id(req.value_id);
    req.kind = MemoryKind::HostExtern;
    plan.alloc_plans[0].push_back(req);
    ET_LOG(
        Debug,
        "[mem] router: alloc-request graph output value_id=%u provider=0 (host) kind=%s mem_id=%d",
        req.value_id,
        to_string(req.kind),
        req.mem_obj_id);
  }

  return Error::Ok;
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch

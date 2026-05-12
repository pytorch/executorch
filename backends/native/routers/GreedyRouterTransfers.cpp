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
#include <optional>
#include <string>

namespace executorch {
namespace backends {
namespace native {
namespace router_internal {

namespace {
using ::executorch::backends::portable::ValueType;
} // namespace

// ---- 8. TransferPlanner: per-segment pre/post transfers + remaps ----
//
// Walks segments in PC order. The MirrorMap is already built (Phase
// 6); this pass only:
//   - Looks up each boundary value's mirror on `cur` (or notes that
//     it's natively bound and needs no mirror).
//   - Records a value_remap so the segment's compiled kernels read
//     from the mirror_id instead of the source vid.
//   - Decides whether a pre-segment upload (host -> mirror) is
//     needed. Always for output boundary (re-aliases per execute);
//     for input boundary, only when the previous writer is on a
//     different runtime (otherwise the mirror still holds fresh
//     bytes from the prior segment).
//   - Decides whether a post-segment download (mirror -> host) is
//     needed: yes if the next reader is on a different runtime, or
//     if v is a graph output with no further reader.
//
// Cross-runtime hops naturally route producer → host → consumer
// (two transfers, both through host) preserving the host-canonical
// invariant — no peer-to-peer transfers ever emitted.
void plan_transfers(RouterContext& ctx) {
  const auto& graph = ctx.graph;

  // segment idx -> (V_orig, V_mirror) pairs to pass as remap.
  ctx.seg_remaps.assign(ctx.segments.size(), {});
  // For each segment, the list of (src_value_id, dst_value_id) transfers.
  // Inserted before the segment's ComputeStep.
  ctx.seg_transfers.assign(ctx.segments.size(), {});
  // Post-Compute transfers (cur -> host): drains a producer's
  // device-side mirror back into the canonical host buffer after the
  // producer's ComputeStep. On UMA (CPU, Apple-Silicon Metal) this
  // SHOULD be a skip-if-same re_alias (no copy). On discrete GPUs
  // (Vulkan) it's a real device->host download.
  ctx.seg_post_transfers.assign(ctx.segments.size(), {});

  // Per-value writer/reader indices in segment order. Built from the
  // (Phase-1-enriched) per-segment input/output sets so in-place
  // mutated args correctly count as both readers AND writers of their
  // backing value_id. Sorted ascending by construction (we iterate
  // segments in order).
  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    for (uint32_t v : ctx.segments[s].output_value_ids) {
      ctx.writers_per_value[v].push_back(s);
    }
    for (uint32_t v : ctx.segments[s].input_value_ids) {
      ctx.readers_per_value[v].push_back(s);
    }
  }

  // -------- Boundary lookup helpers (O(log N)) --------
  auto previous_producer_seg = [&](uint32_t v, size_t s) -> int {
    auto it = ctx.writers_per_value.find(v);
    if (it == ctx.writers_per_value.end())
      return -1;
    const auto& vec = it->second;
    auto lit = std::lower_bound(vec.begin(), vec.end(), s);
    if (lit == vec.begin())
      return -1;
    return static_cast<int>(*(lit - 1));
  };
  auto next_consumer_seg = [&](uint32_t v, size_t s) -> int {
    auto it = ctx.readers_per_value.find(v);
    if (it == ctx.readers_per_value.end())
      return -1;
    const auto& vec = it->second;
    auto uit = std::upper_bound(vec.begin(), vec.end(), s);
    if (uit == vec.end())
      return -1;
    return static_cast<int>(*uit);
  };

  // True iff v is bound natively on runtime p (either homed there or a
  // constant uploaded there) so the segment binds direct, no mirror.
  // Pure check against home + constant consumer info; does NOT inspect
  // alloc_plans state.
  auto value_already_on = [&](uint32_t v, int p) -> bool {
    if (graph.is_constant(v)) {
      auto it = ctx.value_consumer_providers.find(v);
      return it != ctx.value_consumer_providers.end() &&
          it->second.count(p) > 0;
    }
    auto it = ctx.value_home_provider.find(v);
    return it != ctx.value_home_provider.end() && it->second == p;
  };

  // Look up the pre-built mirror for v on dst_p, record the seg_remap
  // for segment s, and return the mirror_value_id (or v itself if v is
  // already natively on dst_p, or nullopt for non-tensor / zero-byte
  // values). Pure lookup — never mints; the MirrorPlanner already
  // produced every (v, dst_p) entry that any segment needs.
  auto lookup_mirror =
      [&](uint32_t v, int dst_p, size_t s) -> std::optional<uint32_t> {
    if (graph.value_type(v) != ValueType::Tensor)
      return std::nullopt;
    if (graph.tensor_nbytes_max(v) == 0)
      return std::nullopt;
    if (value_already_on(v, dst_p))
      return v;

    auto it = ctx.mirror_table.find({v, dst_p});
    if (it == ctx.mirror_table.end())
      return std::nullopt; // shouldn't happen
    uint32_t existing = it->second;

    // Record the seg_remap (deduped — same (v, mirror) pair may
    // already be there from an earlier op in this segment).
    bool already_remapped = false;
    for (const auto& kv : ctx.seg_remaps[s]) {
      if (kv.first == v && kv.second == existing) {
        already_remapped = true;
        break;
      }
    }
    if (!already_remapped) {
      ctx.seg_remaps[s].push_back({v, existing});
    }
    return existing;
  };

  // Look up mirror AND emit a pre-segment upload (host -> mirror) so
  // the segment's compiled kernels see fresh bytes (re-alias semantics
  // for graph IO whose host_ptr changes per execute; data-bringing
  // semantics for cross-runtime intermediates that were just downloaded
  // by their producer).
  auto lookup_mirror_and_upload = [&](uint32_t v, int dst_p, size_t s) {
    // Host-canonical invariant: uploads always target a non-host runtime.
    if (dst_p == kHostIdx)
      return;
    auto m = lookup_mirror(v, dst_p, s);
    if (!m.has_value())
      return;
    if (*m == v)
      return; // natively on dst_p; bind direct, no transfer
    // Constants are immutable + already on every consuming runtime via
    // upload_constants at init; never need per-execute re-upload.
    if (graph.is_constant(v))
      return;
    ctx.seg_transfers[s].push_back(
        {v, *m, /*src_p=*/kHostIdx, /*dst_p=*/dst_p});
    ET_LOG(
        Debug,
        "[mem] router: input-boundary upload seg=%zu v=%u -> mirror=%u (%s)",
        s,
        v,
        *m,
        std::string(ctx.providers[dst_p]->name()).c_str());
  };

  // Emit a post-segment download (cur mirror -> host) draining v's
  // mutated bytes into v's host buffer. The mirror MUST already exist
  // in mirror_table.
  auto ensure_post_download = [&](uint32_t v, int src_p, size_t s) {
    // Host-canonical invariant: downloads always source from a non-host
    // runtime.
    if (src_p == kHostIdx)
      return;
    if (graph.value_type(v) != ValueType::Tensor)
      return;
    if (graph.tensor_nbytes_max(v) == 0)
      return;
    if (graph.is_constant(v))
      return; // constants are immutable
    auto it = ctx.mirror_table.find({v, src_p});
    if (it == ctx.mirror_table.end())
      return; // v is natively on src_p; no mirror to drain
    ctx.seg_post_transfers[s].push_back(
        {it->second, v, /*src_p=*/src_p, /*dst_p=*/kHostIdx});
    ET_LOG(
        Debug,
        "[mem] router: output-boundary download seg=%zu mirror=%u -> v=%u (%s)",
        s,
        it->second,
        v,
        std::string(ctx.providers[src_p]->name()).c_str());
  };

  // -------- Per-segment boundary pass --------
  for (size_t s = 0; s < ctx.segments.size(); ++s) {
    auto& seg = ctx.segments[s];
    int cur = seg.provider_idx;

    // ---- INPUT BOUNDARY: bring each input value onto cur runtime. ----
    for (uint32_t v : seg.input_value_ids) {
      // Constants are pre-uploaded to every consuming runtime at init;
      // the segment binds to the per-runtime constant buffer directly
      // (no remap, no per-execute transfer).
      if (graph.is_constant(v))
        continue;
      // Native binding on cur (homed there or graph IO on host with cur
      // == host): bind direct, no mirror, no transfer.
      if (value_already_on(v, cur))
        continue;

      int prev = previous_producer_seg(v, s);
      int prev_runtime = (prev < 0) ? static_cast<int>(kHostIdx)
                                    : ctx.segments[prev].provider_idx;

      if (prev_runtime == cur) {
        // Same runtime as previous writer: mirror persists across
        // segments via mirror_table — just record the remap so this
        // segment's compiled kernels reference the existing mirror.
        // (No transfer; bytes are still in the mirror from the prior
        // segment's write.)
        lookup_mirror(v, cur, s);
        continue;
      }

      // Cross-runtime: bytes are on host. Bring onto cur via a mirror
      // upload (cur is non-host since value_already_on filtered host).
      lookup_mirror_and_upload(v, cur, s);
    }

    // ---- OUTPUT BOUNDARY: drain each output value back to host where needed.
    // ----
    for (uint32_t v : seg.output_value_ids) {
      if (graph.is_constant(v))
        continue;
      // Native binding on cur: kernel writes directly to v's Buffer on
      // cur. No mirror, no pre-upload, no post-download.
      if (value_already_on(v, cur))
        continue;

      // v is NOT homed on cur. Look up the mirror on cur (already
      // minted by Phase 6) for the kernel to write into.
      auto m = lookup_mirror(v, cur, s);
      if (!m.has_value() || *m == v)
        continue;

      // Pre-segment re-alias upload (host -> cur mirror) so the
      // mirror's underlying Buffer is bound to v's CURRENT host_ptr
      // each execute. Skip-if-same on UMA makes this a no-op when the
      // pointer didn't change. Required for graph IO whose host_ptr
      // changes per bind_outputs / bind_inputs call.
      ctx.seg_transfers[s].push_back(
          {v, *m, /*src_p=*/kHostIdx, /*dst_p=*/cur});
      ET_LOG(
          Debug,
          "[mem] router: output-boundary pre-upload seg=%zu v=%u -> mirror=%u (%s)",
          s,
          v,
          *m,
          std::string(ctx.providers[cur]->name()).c_str());

      // Post-segment download decision based on the next reader's
      // runtime. Same-runtime downstream → no download (mirror persists
      // and the next segment reads from the same mirror). Different-
      // runtime downstream → download (next reader will source from
      // host). No downstream reader → download iff v is a graph output
      // (caller needs the bytes); otherwise the value is dead and the
      // download is wasted.
      int next = next_consumer_seg(v, s);
      bool need_post_download = false;
      if (next < 0) {
        if (ctx.graph_output_ids.count(v) > 0)
          need_post_download = true;
      } else {
        int next_runtime = ctx.segments[next].provider_idx;
        if (next_runtime != cur)
          need_post_download = true;
      }
      if (need_post_download) {
        ensure_post_download(v, cur, s);
      }
    }
  }
}

} // namespace router_internal
} // namespace native
} // namespace backends
} // namespace executorch

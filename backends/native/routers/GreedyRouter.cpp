/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/routers/GreedyRouter.h>

#include <executorch/backends/native/ir/GraphTypes.h>
#include <executorch/backends/native/core/OpDescriptor.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/program_generated.h>

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

namespace {

using ::executorch::backends::portable::Graph;
using ::executorch::backends::portable::OperatorCall;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ValueType = ::executorch::backends::portable::ValueType;

// Pick a provider for an op. Priority: any provider at index >= 1 that
// returns true from can_run wins (host-slot CPU is the fallback).
int pick_provider(const OperatorCall& op, Span<Runtime* const> providers) {
  OpDescriptor desc;
  desc.name = op.name() ? op.name() : "";

  // Try non-host providers first (index 1..N).
  for (size_t i = 1; i < providers.size(); ++i) {
    if (providers[i]->can_run(desc)) {
      return static_cast<int>(i);
    }
  }
  // Fallback: host (index 0) if available.
  if (!providers.empty() && providers[0]->can_run(desc)) {
    return 0;
  }
  return -1;
}

} // namespace

Result<Plan> GreedyRouter::route(
    const Graph& graph,
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

  // ---- 1. Per-instruction provider assignment ---------------------------
  //
  // Walks every instruction in the chain. For Kernel instructions, picks
  // a provider via greedy preference (any non-host provider that accepts
  // wins; CPU is the fallback). For non-Kernel instructions
  // (JumpFalseCall / MoveCall / FreeCall), records them in
  // `control_instrs` for Phase 5 emission and marks the assignment slot
  // as kSkip. DelegateCall is rejected (CONTROL_FLOW_DESIGN.md §11.1).
  //
  // assignments[] is kept parallel to graph instruction indices so the
  // segmenter (Phase 2) can use index arithmetic.
  using ::executorch::backends::portable::InstructionKind;
  using ::executorch::backends::portable::JumpFalseInfo;
  using ::executorch::backends::portable::MoveInfo;
  constexpr int kSkipNonKernel = -2;

  struct ControlInstr {
    uint32_t source_pc;
    InstructionKind kind;
    JumpFalseInfo jf;   // valid iff kind == JumpFalse
    MoveInfo mv;        // valid iff kind == Move
  };
  std::vector<ControlInstr> control_instrs;

  std::vector<int> assignments;
  assignments.reserve(graph.num_instructions());
  for (uint32_t i = 0; i < graph.num_instructions(); ++i) {
    InstructionKind kind = graph.instruction_kind(i);
    if (kind == InstructionKind::Delegate) {
      ET_LOG(
          Error,
          "GreedyRouter: DelegateCall at instr %u — nested delegates "
          "inside the v2 chain are not supported (partitioner contract).",
          i);
      return Error::NotSupported;
    }
    if (kind != InstructionKind::Kernel) {
      assignments.push_back(kSkipNonKernel);
      ControlInstr ci;
      ci.source_pc = i;
      ci.kind = kind;
      if (kind == InstructionKind::JumpFalse) {
        ci.jf = graph.get_jump_false(graph.main_chain_idx(), i);
        ET_LOG(
            Debug,
            "GreedyRouter: instr %u JumpFalseCall(cond=%u, dst=%u)",
            i,
            ci.jf.cond_value_id,
            ci.jf.destination_pc);
      } else if (kind == InstructionKind::Move) {
        ci.mv = graph.get_move(graph.main_chain_idx(), i);
        ET_LOG(
            Debug,
            "GreedyRouter: instr %u MoveCall(src=%u, dst=%u)",
            i,
            ci.mv.src_value_id,
            ci.mv.dst_value_id);
      } else {
        ET_LOG(Debug, "GreedyRouter: instr %u FreeCall (ignored)", i);
      }
      control_instrs.push_back(ci);
      continue;
    }

    OperatorCall op = graph.get_instruction(i);
    int p = pick_provider(op, providers);
    if (p < 0) {
      ET_LOG(
          Error,
          "GreedyRouter: no provider for op '%s'",
          op.name() ? op.name() : "?");
      return Error::NotSupported;
    }
    ET_LOG(
        Debug,
        "GreedyRouter: instr %u op='%s' -> provider %d (%s)",
        i,
        op.name() ? op.name() : "?",
        p,
        std::string(providers[p]->name()).c_str());
    assignments.push_back(p);
  }

  // ---- 2. Group consecutive same-runtime instructions into segments -----
  //
  // CF-2 (CONTROL_FLOW_DESIGN.md §2): segments must NEVER contain a
  // non-Kernel instruction. Every kSkipNonKernel slot is a hard segment
  // break — consecutive kernels around a jump never merge.
  struct PendingSegment {
    int provider_idx;
    std::vector<uint32_t> instruction_indices;
    std::set<uint32_t> input_value_ids; // consumed-but-not-produced
    std::set<uint32_t> output_value_ids; // produced
  };
  std::vector<PendingSegment> segments;

  for (uint32_t i = 0; i < assignments.size(); ++i) {
    if (assignments[i] == kSkipNonKernel) continue; // CF-2 break
    bool start_new = segments.empty()
        || segments.back().provider_idx != assignments[i]
        // Previous instruction was a non-Kernel break.
        || (i > 0 && assignments[i - 1] == kSkipNonKernel);
    if (start_new) {
      segments.push_back({assignments[i], {i}, {}, {}});
    } else {
      segments.back().instruction_indices.push_back(i);
    }
    OperatorCall op = graph.get_instruction(i);
    auto& seg = segments.back();
    for (size_t j = 0; j < op.num_inputs(); ++j) {
      uint32_t v = op.input(j);
      if (seg.output_value_ids.count(v) == 0)
        seg.input_value_ids.insert(v);
    }
    for (size_t j = 0; j < op.num_outputs(); ++j) {
      seg.output_value_ids.insert(op.output(j));
    }
  }
  ET_LOG(
      Debug,
      "GreedyRouter: built %zu segments, %zu control instructions",
      segments.size(),
      control_instrs.size());

  // ---- 3. Build value_id -> producing-segment-index map ----------------
  //
  // value_producer_seg maps value_id → segment index. This is the
  // canonical "who produced this value?" ledger used by DepClosure
  // (Phase 6) and home-provider assignment (Phase 5).
  //
  // EXTENSIBILITY: today values are produced by segments only. With
  // future Region IR (CONTROL_FLOW_DESIGN.md), a value could be
  // produced by an opaque Region (whose internal segment structure
  // the router doesn't introspect). At that point this map's value
  // type may need to widen from "segment_idx" to a tagged union of
  // {segment_idx, region_idx} — or we keep it as segment_idx and add
  // a parallel value_producer_region map. Either is fine; the lambda
  // sites that consume this map (Phase 5 home assignment, Phase 6
  // DepClosure) are already step-kind-aware and would just gain
  // another producer-source case.
  std::unordered_map<uint32_t, size_t> value_producer_seg;
  for (size_t s = 0; s < segments.size(); ++s) {
    for (uint32_t v : segments[s].output_value_ids) {
      value_producer_seg[v] = s;
    }
  }

  // ---- 4. Collect graph IO and value-meta -------------------------------
  std::set<uint32_t> io_ids;
  std::unordered_set<uint32_t> graph_input_ids;
  std::unordered_set<uint32_t> graph_output_ids;
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    uint32_t v = graph.input_id(i);
    io_ids.insert(v);
    graph_input_ids.insert(v);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    uint32_t v = graph.output_id(i);
    io_ids.insert(v);
    graph_output_ids.insert(v);
  }

  // Runtime that consumes each value (could be multiple).
  std::unordered_map<uint32_t, std::set<int>> value_consumer_providers;
  for (auto& seg : segments) {
    for (uint32_t v : seg.input_value_ids) {
      value_consumer_providers[v].insert(seg.provider_idx);
    }
  }

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

  std::unordered_map<uint32_t, int> value_home_provider;
  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor) continue;
    if (graph.tensor_constant_data_key(i) != nullptr) continue; // constant
    if (io_ids.count(i) > 0) {
      value_home_provider[i] = 0; // graph IO → host
      continue;
    }

    // Touching = {producer's runtime if exists} ∪ {consumer runtimes}.
    std::set<int> touching;
    auto pit = value_producer_seg.find(i);
    if (pit != value_producer_seg.end()) {
      touching.insert(segments[pit->second].provider_idx);
    }
    auto cit = value_consumer_providers.find(i);
    if (cit != value_consumer_providers.end()) {
      for (int c : cit->second) touching.insert(c);
    }

    if (touching.empty()) continue; // truly unused
    value_home_provider[i] = (touching.size() == 1) ? *touching.begin() : 0;
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
  std::map<std::pair<uint32_t, int>, uint32_t> mirror_table;
  uint32_t next_mirror_id = static_cast<uint32_t>(graph.num_values());

  auto needs_mirror_on = [&](uint32_t v, int p) -> bool {
    if (graph.value_type(v) != ValueType::Tensor) return false;
    if (graph.tensor_constant_data_key(v) != nullptr) return false;
    auto hit = value_home_provider.find(v);
    if (hit == value_home_provider.end()) return false;
    if (hit->second == p) return false;

    // For graph IO, defer to the engine: it may be able to consume the
    // host pool's IO Buffer directly (e.g., CPU; UMA Metal with
    // newBufferWithBytesNoCopy) and skip the mirror entirely. Default
    // is true (safe), so engines that haven't overridden behave as
    // they do today.
    if (graph_input_ids.count(v)) {
      return instances[p]->wants_input_mirror(v);
    }
    if (graph_output_ids.count(v)) {
      return instances[p]->wants_output_mirror(v);
    }
    // Non-IO cross-runtime intermediate: always mirror.
    return true;
  };

  for (size_t s = 0; s < segments.size(); ++s) {
    int cur = segments[s].provider_idx;
    auto handle = [&](uint32_t v) {
      if (!needs_mirror_on(v, cur)) return;
      auto key = std::make_pair(v, cur);
      if (mirror_table.count(key)) return; // already minted for this (v, cur)
      uint32_t mid = next_mirror_id++;
      mirror_table[key] = mid;
      ET_LOG(
          Debug,
          "[mem] router: minted mirror seg=%zu value_id=%u -> mirror_id=%u (%s)",
          s, v, mid, std::string(providers[cur]->name()).c_str());
    };
    for (uint32_t v : segments[s].input_value_ids) handle(v);
    for (uint32_t v : segments[s].output_value_ids) handle(v);
  }

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
  plan.alloc_plans.assign(providers.size(), {});
  plan.const_plans.assign(providers.size(), {});

  // Helper: compute the MemoryKind to stamp on a host-slot AllocRequest
  // for value v. If any non-host provider produces or consumes v, the
  // host alloc is a HostMirror (will be paired with a device-side
  // DeviceMirror via mirror_values). Otherwise it is HostOnly.
  auto host_kind_for = [&](uint32_t v) -> MemoryKind {
    auto pit = value_producer_seg.find(v);
    if (pit != value_producer_seg.end() &&
        segments[pit->second].provider_idx != 0) {
      return MemoryKind::HostMirror;
    }
    auto cit = value_consumer_providers.find(v);
    if (cit != value_consumer_providers.end()) {
      for (int c : cit->second) {
        if (c != 0) return MemoryKind::HostMirror;
      }
    }
    return MemoryKind::HostOnly;
  };

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
      // Constant. Schedule upload on every consuming provider.
      if (!ndm) {
        ET_LOG(Error, "GreedyRouter: constant '%s' needs NDM", key);
        return Error::InvalidArgument;
      }
      auto it = value_consumer_providers.find(i);
      if (it == value_consumer_providers.end()) {
        // Unused constant — no consumer. Don't emit a ConstRequest;
        // any provider would just waste a Buffer wrapper on bytes
        // nothing reads.
        continue;
      }
      for (int p : it->second) {
        plan.const_plans[p].push_back(
            Engine::ConstRequest{/*value_id=*/i, /*ndm_key=*/key});
        ET_LOG(
            Debug,
            "[mem] router: const-plan value_id=%u key='%s' provider=%d (%s)",
            i,
            key,
            p,
            std::string(providers[p]->name()).c_str());
      }
      continue;
    }

    if (io_ids.count(i) > 0)
      continue; // emitted in graph-IO loop below

    // Intermediate: emit AllocRequest on the value's home provider.
    auto hit = value_home_provider.find(i);
    if (hit == value_home_provider.end())
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
    req.role = BufferRole::Internal;
    req.kind = (home_p == 0) ? host_kind_for(i) : MemoryKind::DeviceOnly;
    plan.alloc_plans[home_p].push_back(req);
    ET_LOG(
        Debug,
        "[mem] router: alloc-request intermediate value_id=%u home_provider=%d (%s) kind=%s mem_id=%d nbytes=%zu",
        i,
        home_p,
        std::string(providers[home_p]->name()).c_str(),
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
        mirror_table.begin(), mirror_table.end());
    std::sort(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    for (const auto& kv : entries) {
      uint32_t v = kv.first.first;
      int dst_p = kv.first.second;
      uint32_t v_mirror = kv.second;
      Engine::AllocRequest req;
      req.value_id = v_mirror;
      req.mem_obj_id = graph.mem_obj_id(v);
      req.host_mirror_value_id = v;
      req.role = BufferRole::Internal;
      req.kind = MemoryKind::DeviceMirror;
      plan.alloc_plans[dst_p].push_back(req);
      plan.mirror_values.push_back({v_mirror, v});
    }
  }

  // Add IO destination requests on the host provider (slot 0). Use
  // graph.mem_obj_id so backends and device-side mirrors stay
  // consistent: if the planner assigned a slot, allocate normally; if
  // not (-1), backends defer to bind_io.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    Engine::AllocRequest req;
    req.value_id = graph.input_id(i);
    req.mem_obj_id = graph.mem_obj_id(req.value_id);
    req.role = BufferRole::Input;
    req.kind = host_kind_for(req.value_id);
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
    req.role = BufferRole::Output;
    req.kind = host_kind_for(req.value_id);
    plan.alloc_plans[0].push_back(req);
    ET_LOG(
        Debug,
        "[mem] router: alloc-request graph output value_id=%u provider=0 (host) kind=%s mem_id=%d",
        req.value_id,
        to_string(req.kind),
        req.mem_obj_id);
  }

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

  // segment idx -> (V_orig, V_mirror) pairs to pass as remap.
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> seg_remaps(
      segments.size());
  // For each segment, the list of (src_value_id, dst_value_id) transfers.
  // Inserted before the segment's ComputeStep.
  struct PendingTransfer {
    uint32_t src_value_id; // source binding
    uint32_t dst_value_id; // destination binding
    int src_provider_idx;
    int dst_provider_idx;
  };
  std::vector<std::vector<PendingTransfer>> seg_transfers(segments.size());
  // Post-Compute transfers (cur -> host): drains a producer's
  // device-side mirror back into the canonical host buffer after the
  // producer's ComputeStep. On UMA (CPU, Apple-Silicon Metal) this
  // SHOULD be a skip-if-same re_alias (no copy). On discrete GPUs
  // (Vulkan) it's a real device->host download.
  std::vector<std::vector<PendingTransfer>> seg_post_transfers(segments.size());

  // Graph inputs/outputs lookup sets (built in Phase 4 above).

  // Per-value writer/reader indices in segment order. Built from the
  // (Phase-1-enriched) per-segment input/output sets so in-place
  // mutated args correctly count as both readers AND writers of their
  // backing value_id. Sorted ascending by construction (we iterate
  // segments in order).
  std::unordered_map<uint32_t, std::vector<size_t>> writers_per_value;
  std::unordered_map<uint32_t, std::vector<size_t>> readers_per_value;
  for (size_t s = 0; s < segments.size(); ++s) {
    for (uint32_t v : segments[s].output_value_ids) {
      writers_per_value[v].push_back(s);
    }
    for (uint32_t v : segments[s].input_value_ids) {
      readers_per_value[v].push_back(s);
    }
  }

  // -------- Boundary lookup helpers (O(log N)) --------
  auto previous_producer_seg = [&](uint32_t v, size_t s) -> int {
    auto it = writers_per_value.find(v);
    if (it == writers_per_value.end()) return -1;
    const auto& vec = it->second;
    auto lit = std::lower_bound(vec.begin(), vec.end(), s);
    if (lit == vec.begin()) return -1;
    return static_cast<int>(*(lit - 1));
  };
  auto next_consumer_seg = [&](uint32_t v, size_t s) -> int {
    auto it = readers_per_value.find(v);
    if (it == readers_per_value.end()) return -1;
    const auto& vec = it->second;
    auto uit = std::upper_bound(vec.begin(), vec.end(), s);
    if (uit == vec.end()) return -1;
    return static_cast<int>(*uit);
  };

  // True iff v is bound natively on runtime p (either homed there or a
  // constant uploaded there) so the segment binds direct, no mirror.
  // Pure check against home + constant consumer info; does NOT inspect
  // alloc_plans state.
  auto value_already_on = [&](uint32_t v, int p) -> bool {
    if (graph.tensor_constant_data_key(v) != nullptr) {
      auto it = value_consumer_providers.find(v);
      return it != value_consumer_providers.end() && it->second.count(p) > 0;
    }
    auto it = value_home_provider.find(v);
    return it != value_home_provider.end() && it->second == p;
  };

  // Look up the pre-built mirror for v on dst_p, record the seg_remap
  // for segment s, and return the mirror_value_id (or v itself if v is
  // already natively on dst_p, or nullopt for non-tensor / zero-byte
  // values). Pure lookup — never mints; the MirrorPlanner already
  // produced every (v, dst_p) entry that any segment needs.
  auto lookup_mirror = [&](uint32_t v, int dst_p, size_t s)
      -> std::optional<uint32_t> {
    if (graph.value_type(v) != ValueType::Tensor) return std::nullopt;
    if (graph.tensor_nbytes_max(v) == 0) return std::nullopt;
    if (value_already_on(v, dst_p)) return v;

    auto it = mirror_table.find({v, dst_p});
    if (it == mirror_table.end()) return std::nullopt; // shouldn't happen
    uint32_t existing = it->second;

    // Record the seg_remap (deduped — same (v, mirror) pair may
    // already be there from an earlier op in this segment).
    bool already_remapped = false;
    for (const auto& kv : seg_remaps[s]) {
      if (kv.first == v && kv.second == existing) {
        already_remapped = true;
        break;
      }
    }
    if (!already_remapped) {
      seg_remaps[s].push_back({v, existing});
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
    if (dst_p == kHostIdx) return;
    auto m = lookup_mirror(v, dst_p, s);
    if (!m.has_value()) return;
    if (*m == v) return; // natively on dst_p; bind direct, no transfer
    // Constants are immutable + already on every consuming runtime via
    // upload_constants at init; never need per-execute re-upload.
    if (graph.tensor_constant_data_key(v) != nullptr) return;
    seg_transfers[s].push_back(
        {v, *m, /*src_p=*/kHostIdx, /*dst_p=*/dst_p});
    ET_LOG(
        Debug,
        "[mem] router: input-boundary upload seg=%zu v=%u -> mirror=%u (%s)",
        s,
        v,
        *m,
        std::string(providers[dst_p]->name()).c_str());
  };

  // Emit a post-segment download (cur mirror -> host) draining v's
  // mutated bytes into v's host buffer. The mirror MUST already exist
  // in mirror_table.
  auto ensure_post_download = [&](uint32_t v, int src_p, size_t s) {
    // Host-canonical invariant: downloads always source from a non-host runtime.
    if (src_p == kHostIdx) return;
    if (graph.value_type(v) != ValueType::Tensor) return;
    if (graph.tensor_nbytes_max(v) == 0) return;
    if (graph.tensor_constant_data_key(v) != nullptr) return; // constants are immutable
    auto it = mirror_table.find({v, src_p});
    if (it == mirror_table.end()) return; // v is natively on src_p; no mirror to drain
    seg_post_transfers[s].push_back(
        {it->second, v, /*src_p=*/src_p, /*dst_p=*/kHostIdx});
    ET_LOG(
        Debug,
        "[mem] router: output-boundary download seg=%zu mirror=%u -> v=%u (%s)",
        s,
        it->second,
        v,
        std::string(providers[src_p]->name()).c_str());
  };

  // -------- Per-segment boundary pass --------
  for (size_t s = 0; s < segments.size(); ++s) {
    auto& seg = segments[s];
    int cur = seg.provider_idx;

    // ---- INPUT BOUNDARY: bring each input value onto cur runtime. ----
    for (uint32_t v : seg.input_value_ids) {
      // Constants are pre-uploaded to every consuming runtime at init;
      // the segment binds to the per-runtime constant buffer directly
      // (no remap, no per-execute transfer).
      if (graph.tensor_constant_data_key(v) != nullptr) continue;
      // Native binding on cur (homed there or graph IO on host with cur
      // == host): bind direct, no mirror, no transfer.
      if (value_already_on(v, cur)) continue;

      int prev = previous_producer_seg(v, s);
      int prev_runtime =
          (prev < 0) ? static_cast<int>(kHostIdx) : segments[prev].provider_idx;

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

    // ---- OUTPUT BOUNDARY: drain each output value back to host where needed. ----
    for (uint32_t v : seg.output_value_ids) {
      if (graph.tensor_constant_data_key(v) != nullptr) continue;
      // Native binding on cur: kernel writes directly to v's Buffer on
      // cur. No mirror, no pre-upload, no post-download.
      if (value_already_on(v, cur)) continue;

      // v is NOT homed on cur. Look up the mirror on cur (already
      // minted by Phase 6) for the kernel to write into.
      auto m = lookup_mirror(v, cur, s);
      if (!m.has_value() || *m == v) continue;

      // Pre-segment re-alias upload (host -> cur mirror) so the
      // mirror's underlying Buffer is bound to v's CURRENT host_ptr
      // each execute. Skip-if-same on UMA makes this a no-op when the
      // pointer didn't change. Required for graph IO whose host_ptr
      // changes per bind_outputs / bind_inputs call.
      seg_transfers[s].push_back(
          {v, *m, /*src_p=*/kHostIdx, /*dst_p=*/cur});
      ET_LOG(
          Debug,
          "[mem] router: output-boundary pre-upload seg=%zu v=%u -> mirror=%u (%s)",
          s,
          v,
          *m,
          std::string(providers[cur]->name()).c_str());

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
        if (graph_output_ids.count(v) > 0) need_post_download = true;
      } else {
        int next_runtime = segments[next].provider_idx;
        if (next_runtime != cur) need_post_download = true;
      }
      if (need_post_download) {
        ensure_post_download(v, cur, s);
      }
    }
  }

  // ---- 7. Compile each segment with its value remap --------------------
  std::vector<CompiledSegment*> compiled_segments;
  for (size_t s = 0; s < segments.size(); ++s) {
    auto& seg = segments[s];
    Engine* inst = instances[seg.provider_idx];
    std::vector<uint32_t> ins(
        seg.input_value_ids.begin(), seg.input_value_ids.end());
    std::vector<uint32_t> outs(
        seg.output_value_ids.begin(), seg.output_value_ids.end());
    ET_LOG(
        Debug,
        "[mem] router: compile_segment %zu provider=%d (%s) instructions=%zu "
        "inputs=%zu outputs=%zu remaps=%zu",
        s,
        seg.provider_idx,
        std::string(providers[seg.provider_idx]->name()).c_str(),
        seg.instruction_indices.size(),
        ins.size(),
        outs.size(),
        seg_remaps[s].size());
    auto r = inst->compile_segment(
        graph,
        Span<const uint32_t>(
            seg.instruction_indices.data(), seg.instruction_indices.size()),
        Span<const uint32_t>(ins.data(), ins.size()),
        Span<const uint32_t>(outs.data(), outs.size()),
        Span<const std::pair<uint32_t, uint32_t>>(
            seg_remaps[s].data(), seg_remaps[s].size()));
    if (!r.ok())
      return r.error();
    compiled_segments.push_back(r.get());
  }

  // ---- 8. Reserve graph input/output bindings --------------------------
  // Input/output destination Buffers are pre-allocated by the executor's
  // allocate_buffers() call (post-route) and bound persistently by
  // prebind_owned_buffers. bind_inputs/bind_outputs re-alias the
  // existing Buffer in place each execute via upload_from_host.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    InputBinding ib;
    ib.value_id = graph.input_id(i);
    plan.inputs.push_back(ib);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    OutputBinding ob;
    ob.value_id = graph.output_id(i);
    plan.outputs.push_back(ob);
  }

  // ---- 9. Emit Steps in order: per-segment transfers, then ComputeStep --
  //
  // Each Step gets a signal event allocated on its owning Engine.
  // wait_for is left empty in v1 (executor walks Steps serially; sync
  // backends settle before the next Step starts). The router records
  // the LAST signal seen on each Engine in plan.terminal_events;
  // the executor waits on those at end of execute() instead of
  // calling drain() per Engine.
  std::vector<Engine*> providers_to_instance(plan.providers.size(), nullptr);
  for (size_t p = 0; p < plan.providers.size() && p < instances.size(); ++p) {
    providers_to_instance[p] = instances[p];
  }
  auto alloc_signal = [&](RuntimeIndex inst_idx) -> EventId {
    Engine* inst = providers_to_instance[inst_idx];
    EventSlot slot;
    slot.event = inst->make_event();
    slot.owner = inst;
    EventId id = static_cast<EventId>(plan.events.size());
    plan.events.push_back(std::move(slot));
    return id;
  };

  // Track the last signal allocated on each Engine — those become
  // terminal events the executor waits on.
  std::unordered_map<RuntimeIndex, EventId> last_signal_per_inst;

  for (size_t s = 0; s < segments.size(); ++s) {
    // source_pc for steps belonging to this segment: the FIRST kernel
    // index in the segment. Used by PCResolver (CONTROL_FLOW_DESIGN.md
    // §8) to map source-PC jump targets to step indices.
    uint32_t seg_first_pc = segments[s].instruction_indices.empty()
        ? kNoSourcePc
        : segments[s].instruction_indices.front();
    for (const auto& xfer : seg_transfers[s]) {
      TransferStep ts;
      ts.src_value_id = xfer.src_value_id;
      ts.dst_value_id = xfer.dst_value_id;
      ts.src_idx = static_cast<RuntimeIndex>(xfer.src_provider_idx);
      ts.dst_idx = static_cast<RuntimeIndex>(xfer.dst_provider_idx);
      ts.source_pc = seg_first_pc;
      // Transfer is issued on the device side (non-host slot of the pair).
      RuntimeIndex issuing =
          (ts.src_idx == 0) ? ts.dst_idx : ts.src_idx;
      ts.signal = alloc_signal(issuing);
      last_signal_per_inst[issuing] = ts.signal;
      plan.steps.emplace_back(std::move(ts));
    }

    ComputeStep cs;
    cs.runtime_idx = static_cast<RuntimeIndex>(segments[s].provider_idx);
    cs.segment = compiled_segments[s];
    cs.source_pc = seg_first_pc;
    cs.signal = alloc_signal(cs.runtime_idx);
    last_signal_per_inst[cs.runtime_idx] = cs.signal;
    plan.steps.emplace_back(std::move(cs));

    for (const auto& xfer : seg_post_transfers[s]) {
      TransferStep ts;
      ts.src_value_id = xfer.src_value_id;
      ts.dst_value_id = xfer.dst_value_id;
      ts.src_idx = static_cast<RuntimeIndex>(xfer.src_provider_idx);
      ts.dst_idx = static_cast<RuntimeIndex>(xfer.dst_provider_idx);
      ts.source_pc = seg_first_pc;
      RuntimeIndex issuing =
          (ts.src_idx == 0) ? ts.dst_idx : ts.src_idx;
      ts.signal = alloc_signal(issuing);
      last_signal_per_inst[issuing] = ts.signal;
      plan.steps.emplace_back(std::move(ts));
    }
  }

  // ---- 10. Control-flow integration (CONTROL_FLOW_DESIGN.md §10) ------
  //
  // Phases:
  //   5. Interleave control instructions (JumpFalseCall / MoveCall) with
  //      the existing source-PC-ordered plan.steps. Emit JumpFalseStep
  //      with kUnresolvedStep, and host-host TransferStep for MoveCall.
  //      PredicateLocator (CONTROL_FLOW_DESIGN.md §6) ensures every
  //      predicate is host-resident.
  //   6. DepClosure (CONTROL_FLOW_DESIGN.md §7) — fill JumpFalseStep
  //      wait_for with the precise transitive signal closure of the
  //      predicate, using value_producer_seg.
  //   7. PCResolver (CONTROL_FLOW_DESIGN.md §8) — second pass mapping
  //      source-PC jump destinations to step indices.
  if (!control_instrs.empty()) {
    // ---- Phase 5a: PredicateLocator -----------------------------------
    //
    // For each JumpFalseCall, decide whether the predicate value is
    // already host-resident. The criterion: value_home_provider[pred]
    // == 0 (host home), OR the value is a graph input / constant /
    // mutable buffer (also host-homed by default). If it isn't, the
    // router must mint a host mirror and emit a download
    // TransferStep BEFORE the JumpFalseStep.
    //
    // Today most predicates come from CPU-pinned prim ops
    // (executorch_prim::eq.Scalar etc.) so the home is already host.
    // We log + error if the predicate is non-host: this is the
    // PredicateLocator §6.3 case-3 path; a follow-up CL implements
    // mirror minting.
    auto predicate_is_host = [&](uint32_t pred_vid) -> bool {
      if (pred_vid >= graph.num_values()) return false;
      // Constant / graph IO / no-producer values default to host.
      if (graph.tensor_constant_data_key(pred_vid) != nullptr) return true;
      if (io_ids.count(pred_vid) > 0) return true;
      auto pit = value_producer_seg.find(pred_vid);
      if (pit == value_producer_seg.end()) return true; // placeholder
      // Producer is host?
      return segments[pit->second].provider_idx == 0
          // Or the value's home is host (so a post-Compute download
          // already exists in plan.steps for it).
          || (value_home_provider.count(pred_vid) > 0
              && value_home_provider[pred_vid] == 0);
    };

    for (const auto& ci : control_instrs) {
      if (ci.kind == InstructionKind::JumpFalse) {
        if (!predicate_is_host(ci.jf.cond_value_id)) {
          ET_LOG(
              Error,
              "GreedyRouter: JumpFalseCall at pc=%u predicate value_id=%u "
              "is not host-resident. Non-host predicates require "
              "PredicateLocator mirror minting (CONTROL_FLOW_DESIGN.md "
              "§6.3 case 3) — not yet supported.",
              ci.source_pc,
              ci.jf.cond_value_id);
          return Error::NotSupported;
        }
      }
    }

    // ---- Phase 5b: Interleave control instructions ---------------------
    //
    // plan.steps is in source-PC order (segment-by-segment). For each
    // source-PC position of a control instruction, insert it AFTER the
    // last step whose source_pc <= ci.source_pc.
    //
    // We rebuild plan.steps via merge.
    std::vector<Step> merged;
    merged.reserve(plan.steps.size() + control_instrs.size() * 2);
    size_t step_i = 0;
    size_t ctrl_i = 0;
    auto step_pc = [&](const Step& s) -> uint32_t {
      if (auto* cs = std::get_if<ComputeStep>(&s)) return cs->source_pc;
      if (auto* ts = std::get_if<TransferStep>(&s)) return ts->source_pc;
      if (auto* jf = std::get_if<JumpFalseStep>(&s)) return jf->source_pc;
      if (auto* ms = std::get_if<MoveStep>(&s)) return ms->source_pc;
      return kNoSourcePc;
    };
    while (step_i < plan.steps.size() || ctrl_i < control_instrs.size()) {
      bool take_step = false;
      if (step_i >= plan.steps.size()) {
        take_step = false;
      } else if (ctrl_i >= control_instrs.size()) {
        take_step = true;
      } else {
        uint32_t s_pc = step_pc(plan.steps[step_i]);
        uint32_t c_pc = control_instrs[ctrl_i].source_pc;
        // Existing steps come BEFORE a control instruction at the same
        // PC: a JumpFalseCall / MoveCall logically follows the steps
        // that produced its operands (those steps' source_pc is the
        // segment's first PC, which is <= the control instruction's PC
        // because the segment ended right before the control instr).
        take_step = (s_pc != kNoSourcePc && s_pc <= c_pc);
      }

      if (take_step) {
        merged.push_back(std::move(plan.steps[step_i++]));
      } else {
        const auto& ci = control_instrs[ctrl_i++];
        if (ci.kind == InstructionKind::JumpFalse) {
          JumpFalseStep jf;
          jf.pred_value_id = ci.jf.cond_value_id;
          jf.dst_step_idx = kUnresolvedStep;
          jf.unresolved_dst_pc = ci.jf.destination_pc;
          jf.source_pc = ci.source_pc;
          // wait_for is filled by Phase 6 (DepClosure) below.
          merged.emplace_back(std::move(jf));
        } else if (ci.kind == InstructionKind::Move) {
          // EValue-level assignment, NOT a byte copy. Distinct from
          // TransferStep (which is for cross-runtime byte movement).
          // See CONTROL_FLOW_DESIGN.md §16.
          MoveStep ms;
          ms.src_value_id = ci.mv.src_value_id;
          ms.dst_value_id = ci.mv.dst_value_id;
          ms.source_pc = ci.source_pc;
          merged.emplace_back(std::move(ms));
        } else if (ci.kind == InstructionKind::Free) {
          // Ignored: v2 memory plan handles deallocation. See
          // CONTROL_FLOW_DESIGN.md §11.1, §15.
          continue;
        }
      }
    }
    plan.steps = std::move(merged);

    // ---- Phase 6: DepClosure -------------------------------------------
    //
    // For each value_id, determine which step (if any) makes it
    // host-visible: a host ComputeStep that produces it, or a
    // TransferStep with dst on host, or a MoveStep that aliases it
    // from a host-visible source. This is the basis for closure walks.
    //
    // EXTENSIBILITY: when adding a new Step variant (e.g., a future
    // RegionStep that delegates a whole control-flow region to one
    // Engine), three lambdas below need updates:
    //
    //   * host_visible_producer: which value_ids does this step make
    //     visible on host? (For RegionStep on host: all the region's
    //     declared outputs.)
    //   * step_input_values: which value_ids does this step consume?
    //     (For RegionStep: all values referenced by the region's
    //     predicate-cone + branch bodies that originate outside it.)
    //   * step_signal: what signal does this step emit, if any?
    //
    // Update all three at the same time. DepClosure traversal works
    // automatically once these lambdas know about the new variant.
    std::unordered_map<uint32_t, size_t> host_visible_producer;
    for (size_t si = 0; si < plan.steps.size(); ++si) {
      const auto& s = plan.steps[si];
      if (auto* cs = std::get_if<ComputeStep>(&s)) {
        if (cs->runtime_idx != 0) continue; // non-host; not directly visible
        // The segment with this source_pc produces these outputs.
        // segments was indexed by segment number, not source_pc; find
        // the segment that starts at cs->source_pc.
        for (const auto& seg : segments) {
          if (!seg.instruction_indices.empty()
              && seg.instruction_indices.front() == cs->source_pc) {
            for (uint32_t v : seg.output_value_ids) {
              host_visible_producer[v] = si;
            }
            break;
          }
        }
      } else if (auto* ts = std::get_if<TransferStep>(&s)) {
        if (ts->dst_idx == 0) {
          host_visible_producer[ts->dst_value_id] = si;
        }
      } else if (auto* ms = std::get_if<MoveStep>(&s)) {
        // EValue assignment: dst becomes host-visible, dependent on src.
        host_visible_producer[ms->dst_value_id] = si;
      }
    }

    // For each step, the set of value_ids whose visibility it consumes
    // (so we can recurse). For ComputeStep: its segment.input_value_ids.
    // For TransferStep host-bound: its src_value_id.
    // For MoveStep: its src_value_id.
    auto step_input_values = [&](size_t si) -> std::vector<uint32_t> {
      std::vector<uint32_t> out;
      const auto& s = plan.steps[si];
      if (auto* cs = std::get_if<ComputeStep>(&s)) {
        for (const auto& seg : segments) {
          if (!seg.instruction_indices.empty()
              && seg.instruction_indices.front() == cs->source_pc) {
            out.assign(
                seg.input_value_ids.begin(), seg.input_value_ids.end());
            break;
          }
        }
      } else if (auto* ts = std::get_if<TransferStep>(&s)) {
        out.push_back(ts->src_value_id);
      } else if (auto* ms = std::get_if<MoveStep>(&s)) {
        out.push_back(ms->src_value_id);
      }
      return out;
    };

    auto step_signal = [&](size_t si) -> EventId {
      const auto& s = plan.steps[si];
      if (auto* cs = std::get_if<ComputeStep>(&s)) return cs->signal;
      if (auto* ts = std::get_if<TransferStep>(&s)) return ts->signal;
      // MoveStep / JumpFalseStep have no signal (host-synchronous).
      return kNoEvent;
    };

    auto dep_closure = [&](uint32_t pred_vid) -> std::vector<EventId> {
      std::unordered_set<EventId> sigs;
      std::unordered_set<uint32_t> visited;
      std::vector<uint32_t> work = {pred_vid};
      while (!work.empty()) {
        uint32_t v = work.back();
        work.pop_back();
        if (!visited.insert(v).second) continue;
        auto pit = host_visible_producer.find(v);
        if (pit == host_visible_producer.end()) continue;
        EventId sig = step_signal(pit->second);
        if (sig != kNoEvent) sigs.insert(sig);
        for (uint32_t in : step_input_values(pit->second)) work.push_back(in);
      }
      return std::vector<EventId>(sigs.begin(), sigs.end());
    };

    for (auto& s : plan.steps) {
      if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
        jf->wait_for = dep_closure(jf->pred_value_id);
        ET_LOG(
            Debug,
            "[cf] DepClosure pc=%u pred_vid=%u -> %zu signals",
            jf->source_pc,
            jf->pred_value_id,
            jf->wait_for.size());
      }
    }

    // ---- Phase 7: PCResolver ------------------------------------------
    //
    // Source-PC → step-index map, second pass. Walk steps in order;
    // each step records its lowest source PC. Then backward-fill so
    // that any source PC inherits the step at or after it.
    size_t n_instr = graph.num_instructions();
    std::vector<size_t> pc_to_step(n_instr, kUnresolvedStep);
    for (size_t si = 0; si < plan.steps.size(); ++si) {
      uint32_t pc = step_pc(plan.steps[si]);
      if (pc == kNoSourcePc || pc >= n_instr) continue;
      // Keep the earliest step at this PC.
      if (pc_to_step[pc] == kUnresolvedStep || pc_to_step[pc] > si) {
        pc_to_step[pc] = si;
      }
    }
    // Backward fill.
    size_t next_step = plan.steps.size();
    for (size_t pc_rev = 0; pc_rev < n_instr; ++pc_rev) {
      size_t pc = n_instr - 1 - pc_rev;
      if (pc_to_step[pc] != kUnresolvedStep) {
        next_step = pc_to_step[pc];
      } else {
        pc_to_step[pc] = next_step;
      }
    }
    // Resolve every JumpFalseStep.
    for (auto& s : plan.steps) {
      if (auto* jf = std::get_if<JumpFalseStep>(&s)) {
        if (jf->unresolved_dst_pc < n_instr) {
          jf->dst_step_idx = pc_to_step[jf->unresolved_dst_pc];
        } else {
          // Destination is past the end → falls off the end of the
          // chain (legitimate: branch-end jumps).
          jf->dst_step_idx = plan.steps.size();
        }
        ET_LOG(
            Debug,
            "[cf] PCResolver pc=%u dst_pc=%u -> step_idx=%zu",
            jf->source_pc,
            jf->unresolved_dst_pc,
            jf->dst_step_idx);
      }
    }
  }

  // Terminal events = last signal on each Engine.
  for (const auto& kv : last_signal_per_inst) {
    plan.terminal_events.push_back(kv.second);
  }

  if (options.dump_trace) {
    ET_LOG(
        Info,
        "GreedyRouter: %zu segments / %zu steps / %zu mirror values",
        segments.size(),
        plan.steps.size(),
        plan.mirror_values.size());
  }

  return plan;
}

} // namespace native
} // namespace backends
} // namespace executorch

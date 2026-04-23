/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/portable/runtime_v2/routers/GreedyRouter.h>

#include <executorch/backends/portable/runtime_v2/api/OpDescriptor.h>
#include <executorch/backends/portable/runtime_v2/api/GraphTypes.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/program_generated.h>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

namespace {

using ::executorch::backends::portable::Graph;
using ::executorch::backends::portable::OperatorCall;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;
using ValueType = ::executorch::backends::portable::ValueType;

// Pick a provider for an op. Priority: any provider at index >= 1 that
// returns true from can_run wins (host-slot CPU is the fallback).
int pick_provider(const OperatorCall& op, Span<Provider* const> providers) {
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

}  // namespace

Result<Plan> GreedyRouter::route(
    const Graph& graph,
    Span<Provider* const> providers,
    Span<Instance* const> instances,
    const ::executorch::runtime::NamedDataMap* ndm,
    const RouterOptions& options) {
  if (providers.size() != instances.size()) return Error::InvalidArgument;
  if (providers.empty()) return Error::InvalidArgument;

  Plan plan;
  plan.providers.assign(providers.begin(), providers.end());
  plan.instances.assign(instances.begin(), instances.end());

  // ---- 1. Per-instruction provider assignment ---------------------------
  std::vector<int> assignments;
  assignments.reserve(graph.num_instructions());
  for (uint32_t i = 0; i < graph.num_instructions(); ++i) {
    OperatorCall op = graph.get_instruction(i);
    int p = pick_provider(op, providers);
    if (p < 0) {
      ET_LOG(Error, "GreedyRouter: no provider for op '%s'",
             op.name() ? op.name() : "?");
      return Error::NotSupported;
    }
    ET_LOG(Debug, "GreedyRouter: instr %u op='%s' -> provider %d (%s)", i,
           op.name() ? op.name() : "?", p,
           std::string(providers[p]->name()).c_str());
    assignments.push_back(p);
  }

  // ---- 2. Group consecutive same-runtime instructions into segments -----
  struct PendingSegment {
    int provider_idx;
    std::vector<uint32_t> instruction_indices;
    std::set<uint32_t> input_value_ids;   // consumed-but-not-produced
    std::set<uint32_t> output_value_ids;  // produced
  };
  std::vector<PendingSegment> segments;

  if (!assignments.empty()) {
    segments.push_back({assignments[0], {0}, {}, {}});
  }
  for (uint32_t i = 0; i < assignments.size(); ++i) {
    if (i > 0 && assignments[i] != segments.back().provider_idx) {
      segments.push_back({assignments[i], {i}, {}, {}});
    } else if (i > 0) {
      segments.back().instruction_indices.push_back(i);
    }
    OperatorCall op = graph.get_instruction(i);
    auto& seg = segments.back();
    for (size_t j = 0; j < op.num_inputs(); ++j) {
      uint32_t v = op.input(j);
      if (seg.output_value_ids.count(v) == 0) seg.input_value_ids.insert(v);
    }
    for (size_t j = 0; j < op.num_outputs(); ++j) {
      seg.output_value_ids.insert(op.output(j));
    }
  }
  ET_LOG(Debug, "GreedyRouter: built %zu segments", segments.size());

  // ---- 3. Build value_id -> producing-segment-index map ----------------
  std::unordered_map<uint32_t, size_t> value_producer_seg;
  for (size_t s = 0; s < segments.size(); ++s) {
    for (uint32_t v : segments[s].output_value_ids) {
      value_producer_seg[v] = s;
    }
  }

  // ---- 4. Collect graph IO and value-meta -------------------------------
  std::set<uint32_t> io_ids;
  for (size_t i = 0; i < graph.num_input_ids(); ++i)
    io_ids.insert(graph.input_id(i));
  for (size_t i = 0; i < graph.num_output_ids(); ++i)
    io_ids.insert(graph.output_id(i));

  // Provider that consumes each value (could be multiple).
  std::unordered_map<uint32_t, std::set<int>> value_consumer_providers;
  for (auto& seg : segments) {
    for (uint32_t v : seg.input_value_ids) {
      value_consumer_providers[v].insert(seg.provider_idx);
    }
  }

  // ---- 5. Decide where each value's "home" Buffer lives ----------------
  // Unified rule:
  //   - Graph IO                    → home = host (provider 0).
  //   - Aliased group (same mem_obj_id, set by AOT memory planner): if
  //     ALL ops touching ANY value in the group are on a single non-host
  //     runtime, home = that runtime (e.g. KV cache stays on Vulkan when
  //     all ops including the writeback are on Vulkan). Otherwise home =
  //     host (correctness via cross-runtime mirrors; perf hit acceptable).
  //   - Cross-runtime intermediate  → home = host.
  //   - Otherwise (single-runtime)  → home = producer's runtime.
  //
  // Allocating cross-runtime values on host means every synthetic mirror
  // sources from host, so allocate_buffers' host-first iteration order
  // naturally satisfies dependencies — no two-pass needed.

  // Build mem_obj_id groups (only for values with mem_obj_id >= 0; the
  // sort-and-index Graph::mem_obj_id assigns -1 for non-tensor / no-alloc
  // values, and a unique non-negative id per (pool, offset) slot).
  std::unordered_map<int32_t, std::vector<uint32_t>> mem_obj_groups;
  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor) continue;
    if (graph.tensor_constant_data_key(i) != nullptr) continue;
    int32_t mid = graph.mem_obj_id(i);
    if (mid >= 0) mem_obj_groups[mid].push_back(i);
  }

  // Identify SEMANTIC alias mem_obj_ids: groups that include a mutable
  // buffer placeholder (value with allocation info, not graph IO, not
  // constant, not produced by any op). These come from tag_mutated_buffer
  // + AOT spec-sharing: the placeholder and the mutation source share a
  // mem_obj_id by design, so writing to the source IS writing to the
  // buffer — they MUST share storage at runtime.
  //
  // Other mem_obj_id groups (e.g. lifetime reuse where the planner
  // packed two values with non-overlapping lifetimes into one slot)
  // don't have this constraint — they're aliased in storage but not
  // semantically the same data; per-value home rules suffice.
  std::set<int32_t> semantic_alias_mids;
  for (size_t i = 0; i < graph.num_mutable_buffer_ids(); ++i) {
    uint32_t buf_vid = graph.mutable_buffer_id(i);
    int32_t mid = graph.mem_obj_id(buf_vid);
    if (mid >= 0) semantic_alias_mids.insert(mid);
  }

  // Helper: union of all touching runtimes for one value_id.
  auto touching_runtimes = [&](uint32_t v) -> std::set<int> {
    std::set<int> rts;
    auto pit = value_producer_seg.find(v);
    if (pit != value_producer_seg.end()) {
      rts.insert(segments[pit->second].provider_idx);
    }
    auto cit = value_consumer_providers.find(v);
    if (cit != value_consumer_providers.end()) {
      for (int p : cit->second) rts.insert(p);
    }
    return rts;
  };

  std::unordered_map<uint32_t, int> value_home_provider;
  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor) continue;
    if (graph.tensor_constant_data_key(i) != nullptr) continue;  // constant
    if (io_ids.count(i) > 0) {
      value_home_provider[i] = 0;  // graph IO → host
      continue;
    }

    // Aliased-group rule: applies ONLY to semantic alias groups (where
    // the planner deliberately shares storage between a buffer placeholder
    // and its mutation source). Lifetime-reuse aliasing falls through to
    // per-value rules.
    int32_t mid = graph.mem_obj_id(i);
    if (mid >= 0 && semantic_alias_mids.count(mid)) {
      auto git = mem_obj_groups.find(mid);
      if (git != mem_obj_groups.end() && git->second.size() > 1) {
        std::set<int> all_rts;
        for (uint32_t v : git->second) {
          for (int p : touching_runtimes(v)) all_rts.insert(p);
        }
        // Single non-host runtime touches all members → home there
        // (fast path: zero cross-runtime traffic).
        if (all_rts.size() == 1 && *all_rts.begin() != 0) {
          value_home_provider[i] = *all_rts.begin();
        } else {
          // Mixed runtimes (or all on host) → home = host. Correctness
          // via cross-runtime mirrors; non-unified backends (Vulkan)
          // pay upload+download per execute.
          value_home_provider[i] = 0;
        }
        continue;
      }
    }

    // Non-aliased: per-value rule.
    auto pit = value_producer_seg.find(i);
    auto cit = value_consumer_providers.find(i);

    if (pit != value_producer_seg.end()) {
      // Has a producer in the delegate (intermediate). Cross-runtime if
      // any cross-segment consumer is on a different runtime than the
      // producer.
      int producer_p = segments[pit->second].provider_idx;
      bool cross_runtime = false;
      if (cit != value_consumer_providers.end()) {
        for (int c : cit->second) {
          if (c != producer_p) { cross_runtime = true; break; }
        }
      }
      value_home_provider[i] = cross_runtime ? 0 : producer_p;
    } else {
      // No producer = placeholder (mutable buffer pulled into delegate
      // by tag_mutated_buffer; not a graph input). Home = consumer's
      // runtime if single-runtime consumer set, else host.
      if (cit == value_consumer_providers.end()) continue;  // truly unused
      const std::set<int>& consumers = cit->second;
      if (consumers.size() == 1) {
        value_home_provider[i] = *consumers.begin();
      } else {
        value_home_provider[i] = 0;
      }
    }
  }

  // ---- 6. Upload constants and emit alloc-plans for intermediates -----
  // Constants are uploaded to ALL providers that consume them (synchronous
  // upload_constant call; not subject to allocate_all). For each
  // (provider, value_id, mem_obj_id) tuple we dedupe via a map.
  // Intermediates are NOT allocated here — the router emits an
  // AllocRequest entry into plan.alloc_plans[home_provider]; the
  // executor's allocate_buffers step (called after route + materialize)
  // performs the actual allocation via allocate_all.

  // Initialize alloc_plans (one entry per provider).
  plan.alloc_plans.assign(providers.size(), {});

  // (provider_idx, mem_obj_id) -> sentinel to track if mem_obj_id already
  // emitted for this provider (so we emit at most one AllocRequest per
  // (provider, mem_obj_id) group).
  std::set<std::pair<int, int32_t>> mem_id_emitted;

  for (uint32_t i = 0; i < graph.num_values(); ++i) {
    if (graph.value_type(i) != ValueType::Tensor) continue;

    if (const char* key = graph.tensor_constant_data_key(i); key != nullptr) {
      // Constant. Upload to each consuming provider via upload_constant
      // (separate path from allocate_all; persistent zero-copy alias).
      if (!ndm) {
        ET_LOG(Error, "GreedyRouter: constant '%s' needs NDM", key);
        return Error::InvalidArgument;
      }
      auto it = value_consumer_providers.find(i);
      std::set<int> consumers =
          it != value_consumer_providers.end() ? it->second : std::set<int>{0};
      for (int p : consumers) {
        auto buf_result = instances[p]->upload_constant(*ndm, key);
        if (!buf_result.ok()) {
          ET_LOG(Error, "GreedyRouter: upload_constant('%s') on provider %d failed",
                 key, p);
          return buf_result.error();
        }
        ET_LOG(Debug,
               "[mem] router: upload_constant value_id=%u key='%s' provider=%d (%s) bytes=%zu",
               i, key, p,
               std::string(providers[p]->name()).c_str(),
               buf_result.get()->size_bytes());
        if (p == *consumers.begin()) {
          plan.owned_buffers.push_back({buf_result.get(), instances[p], i});
        }
      }
      continue;
    }

    if (io_ids.count(i) > 0) continue;  // emitted in graph-IO loop below

    // Intermediate: emit AllocRequest on the value's home provider.
    auto hit = value_home_provider.find(i);
    if (hit == value_home_provider.end()) continue;  // unused
    int home_p = hit->second;

    int32_t mem_id = graph.mem_obj_id(i);
    size_t nbytes = graph.tensor_nbytes_max(i);
    if (nbytes == 0) continue;

    // Always emit an AllocRequest per value_id. Even if mem_id is
    // shared with another value (e.g., AOT memory planner aliased them
    // because their lifetimes don't overlap, or because of an op like
    // aten::copy_'s formal `out` sharing data with `src`), each value_id
    // gets its own binding entry. Backends that want to honor mem_id
    // sharing as actual storage aliasing (Vulkan SharedObject style)
    // can do so internally based on req.mem_obj_id; for our current
    // backends each value_id gets its own Buffer.
    Instance::AllocRequest req;
    req.value_id = i;
    req.mem_obj_id = mem_id;
    req.host_alias = nullptr;
    plan.alloc_plans[home_p].push_back(req);
    ET_LOG(Debug,
           "[mem] router: alloc-request intermediate value_id=%u home_provider=%d (%s) mem_id=%d nbytes=%zu",
           i, home_p,
           std::string(providers[home_p]->name()).c_str(),
           mem_id, nbytes);
  }

  // Add IO destination requests on the host provider (slot 0). bind_inputs
  // / bind_outputs will re-alias these to caller storage per execute.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    Instance::AllocRequest req;
    req.value_id = graph.input_id(i);
    req.mem_obj_id = -1;
    req.host_alias = nullptr;
    plan.alloc_plans[0].push_back(req);
    ET_LOG(Debug,
           "[mem] router: alloc-request graph input value_id=%u provider=0 (host)",
           req.value_id);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    Instance::AllocRequest req;
    req.value_id = graph.output_id(i);
    req.mem_obj_id = -1;
    req.host_alias = nullptr;
    plan.alloc_plans[0].push_back(req);
    ET_LOG(Debug,
           "[mem] router: alloc-request graph output value_id=%u provider=0 (host)",
           req.value_id);
  }

  // ---- 6. For each cross-segment value, synthesize destination ----------
  // For each segment's input value V where the producing segment's provider
  // differs from this segment's provider:
  //   - Synthesize a destination value_id V_synth on dst_p.
  //   - Emit an AllocRequest for V_synth into plan.alloc_plans[dst_p].
  //     The executor's allocate_buffers step will patch host_alias to
  //     point at V's Buffer (after V is allocated on src_p) so the
  //     destination backend can choose to zero-copy alias the host
  //     pointer (CPU/Apple-Silicon Metal) or allocate fresh + copy
  //     per-execute (Vulkan).
  //   - Emit a TransferStep V → V_synth (always — per-execute work is
  //     a no-op skip-if-same on host-addressable runtimes).
  //   - Record V → V_synth remap for the destination CompiledSegment.

  uint32_t next_synth_id = static_cast<uint32_t>(graph.num_values());
  // segment idx -> (V_orig, V_synth) pairs to pass as remap.
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> seg_remaps(
      segments.size());
  // For each segment, the list of (src_value_id, dst_value_id) transfers.
  // Inserted before the segment's ComputeStep.
  struct PendingTransfer {
    uint32_t src_value_id;  // V (producer's binding)
    uint32_t dst_value_id;  // V' (this segment's view)
    int src_provider_idx;
    int dst_provider_idx;
  };
  std::vector<std::vector<PendingTransfer>> seg_transfers(segments.size());

  for (size_t s = 0; s < segments.size(); ++s) {
    auto& seg = segments[s];
    int dst_p = seg.provider_idx;
    for (uint32_t v : seg.input_value_ids) {
      auto pit = value_producer_seg.find(v);
      if (pit == value_producer_seg.end()) {
        // Graph input or constant — handled by bind_inputs / step 5.
        continue;
      }
      int src_p = segments[pit->second].provider_idx;
      if (src_p == dst_p) continue;  // same provider, no transfer

      // Dedup: if v already has an allocation on dst_p (e.g., it's a
      // graph output whose host alloc was emitted in step 5, and dst_p
      // is host), the consumer can read v directly from its existing
      // Buffer. Skipping the redundant mirror is also REQUIRED for
      // correctness here: if v is allocated in dst_p's plan AND we
      // emit a synthetic mirror also in dst_p's plan sourcing from v,
      // allocate_buffers' host_alias patching can't resolve the source
      // (the source is in the same provider's plan and isn't in the
      // value_to_buf ledger until that plan's allocate_all returns).
      bool v_already_on_dst = false;
      for (const auto& req : plan.alloc_plans[dst_p]) {
        if (req.value_id == v) { v_already_on_dst = true; break; }
      }
      if (v_already_on_dst) {
        ET_LOG(Debug,
               "[mem] router: skip mirror for value_id=%u in seg=%zu "
               "(already allocated on provider=%d)",
               v, s, dst_p);
        continue;
      }

      // Only tensors get transferred / aliased.
      if (graph.value_type(v) != ValueType::Tensor) continue;
      size_t nbytes = graph.tensor_nbytes_max(v);
      if (nbytes == 0) continue;

      uint32_t v_synth = next_synth_id++;

      // Emit AllocRequest for the synthetic value on dst_p.
      // host_alias is left null here; the executor patches it to point
      // at v's Buffer (looked up from already-allocated src_p) before
      // calling allocate_all. mem_obj_id = -1 (synthetic dedicated).
      Instance::AllocRequest req;
      req.value_id = v_synth;
      req.mem_obj_id = -1;
      req.host_alias = nullptr;  // patched by allocate_buffers
      plan.alloc_plans[dst_p].push_back(req);

      plan.synthetic_values.push_back({v_synth, v});
      seg_remaps[s].push_back({v, v_synth});
      seg_transfers[s].push_back({v, v_synth, src_p, dst_p});

      ET_LOG(Debug,
             "[mem] router: cross-runtime mirror seg=%zu value_id=%u (%s) -> synth_id=%u (%s) bytes=%zu",
             s, v, std::string(providers[src_p]->name()).c_str(),
             v_synth, std::string(providers[dst_p]->name()).c_str(),
             nbytes);
    }
  }

  // ---- 6b. Producer-side mirror for any value whose home is host but
  //          whose producing segment is on a non-host provider. -------
  // Includes:
  //   - Graph outputs produced by a non-host segment.
  //   - Cross-runtime intermediates (homed on host by step 5).
  // For each such value V:
  //   - Synthesize V_synth on producer's runtime, source = V's host alloc.
  //   - Add seg_remap so producer kernel writes to V_synth (which
  //     host_aliases V's host buffer; bytes land in host memory directly
  //     on Apple Silicon Metal — zero copy).
  //   - Emit TransferStep host -> producer BEFORE producer's ComputeStep
  //     so V_synth's Metal Buffer re-aliases V's current host_ptr per
  //     execute (matches bind_outputs / bind_inputs rebinding).
  //
  // This is the symmetric counterpart of the consumer-side mirror loop
  // above. Together they handle all cross-runtime data flow with the
  // same machinery; the value's "home" decides which side gets the mirror.
  for (uint32_t v : [&]() {
         std::vector<uint32_t> vs;
         for (const auto& kv : value_home_provider) {
           if (kv.second == 0) vs.push_back(kv.first);
         }
         std::sort(vs.begin(), vs.end());
         return vs;
       }()) {
    auto pit = value_producer_seg.find(v);
    if (pit == value_producer_seg.end()) continue;  // graph input (no producer in delegate)
    int producer_p = segments[pit->second].provider_idx;
    if (producer_p == 0) continue;  // producer IS host; writes directly
    if (graph.value_type(v) != ValueType::Tensor) continue;

    uint32_t v_synth = next_synth_id++;
    Instance::AllocRequest req;
    req.value_id = v_synth;
    req.mem_obj_id = -1;
    req.host_alias = nullptr;  // patched by allocate_buffers from v's host Buffer
    plan.alloc_plans[producer_p].push_back(req);

    plan.synthetic_values.push_back({v_synth, v});
    seg_remaps[pit->second].push_back({v, v_synth});
    // Per-execute re-alias: host -> producer mirror, runs BEFORE the
    // producing segment's ComputeStep.
    seg_transfers[pit->second].push_back(
        {v, v_synth, /*src_p=*/0, /*dst_p=*/producer_p});

    ET_LOG(Debug,
           "[mem] router: producer-side mirror value_id=%u (host) -> "
           "synth_id=%u (%s) for producing seg=%zu",
           v, v_synth,
           std::string(providers[producer_p]->name()).c_str(),
           pit->second);
  }

  // ---- 7. Compile each segment with its value remap --------------------
  std::vector<CompiledSegment*> compiled_segments;
  for (size_t s = 0; s < segments.size(); ++s) {
    auto& seg = segments[s];
    Instance* inst = instances[seg.provider_idx];
    std::vector<uint32_t> ins(seg.input_value_ids.begin(),
                              seg.input_value_ids.end());
    std::vector<uint32_t> outs(seg.output_value_ids.begin(),
                               seg.output_value_ids.end());
    ET_LOG(Debug,
           "[mem] router: compile_segment %zu provider=%d (%s) instructions=%zu "
           "inputs=%zu outputs=%zu remaps=%zu",
           s, seg.provider_idx,
           std::string(providers[seg.provider_idx]->name()).c_str(),
           seg.instruction_indices.size(), ins.size(),
           outs.size(), seg_remaps[s].size());
    auto r = inst->compile_segment(
        graph,
        Span<const uint32_t>(seg.instruction_indices.data(),
                             seg.instruction_indices.size()),
        Span<const uint32_t>(ins.data(), ins.size()),
        Span<const uint32_t>(outs.data(), outs.size()),
        Span<const std::pair<uint32_t, uint32_t>>(seg_remaps[s].data(),
                                                  seg_remaps[s].size()));
    if (!r.ok()) return r.error();
    compiled_segments.push_back(r.get());
  }

  // ---- 8. Reserve graph input/output bindings --------------------------
  // Input/output destination Buffers are pre-allocated by the executor's
  // allocate_buffers() call (post-route) and bound persistently by
  // prebind_owned_buffers. bind_inputs/bind_outputs re-alias the
  // existing Buffer in place each execute via upload_from_host.
  for (size_t i = 0; i < graph.num_input_ids(); ++i) {
    InputBinding ib;
    ib.loc = Location::host();
    ib.value_id = graph.input_id(i);
    plan.inputs.push_back(ib);
  }
  for (size_t i = 0; i < graph.num_output_ids(); ++i) {
    OutputBinding ob;
    ob.loc = Location::host();
    ob.value_id = graph.output_id(i);
    plan.outputs.push_back(ob);
  }

  // ---- 9. Emit Steps in order: per-segment transfers, then ComputeStep --
  for (size_t s = 0; s < segments.size(); ++s) {
    for (const auto& xfer : seg_transfers[s]) {
      TransferStep ts;
      ts.src_value_id = xfer.src_value_id;
      ts.dst_value_id = xfer.dst_value_id;
      ts.src = (xfer.src_provider_idx == 0)
                   ? Location::host()
                   : Location::on(providers[xfer.src_provider_idx]->id());
      ts.dst = (xfer.dst_provider_idx == 0)
                   ? Location::host()
                   : Location::on(providers[xfer.dst_provider_idx]->id());
      ts.src_idx = static_cast<RuntimeIndex>(xfer.src_provider_idx);
      ts.dst_idx = static_cast<RuntimeIndex>(xfer.dst_provider_idx);
      ts.queue = QueueKind::Transfer;
      ts.signal = kNoEvent;
      plan.steps.emplace_back(ts);
    }

    ComputeStep cs;
    cs.runtime_idx = static_cast<RuntimeIndex>(segments[s].provider_idx);
    cs.segment = compiled_segments[s];
    cs.queue = QueueKind::Compute;
    cs.signal = kNoEvent;
    plan.steps.emplace_back(cs);
  }

  if (options.dump_trace) {
    ET_LOG(Info,
           "GreedyRouter: %zu segments / %zu steps / %zu owned buffers / "
           "%zu synthetic values",
           segments.size(), plan.steps.size(), plan.owned_buffers.size(),
           plan.synthetic_values.size());
  }

  return plan;
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

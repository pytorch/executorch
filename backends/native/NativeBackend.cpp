/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * NativeBackend — ExecuTorch BackendInterface adapter for the v8.2
 * portable runtime architecture.
 *
 * Registers "NativeBackend" with the runtime, separate from the
 * existing "PortableBackend" so the two can coexist during migration.
 *
 * See PORTABLE_BACKEND_API_PROPOSAL.md (§3, §6) for the architecture.
 *
 * This file is the slim entry point. The init/execute helpers live in
 * sibling translation units, all sharing types via NativeBackendInternal.h:
 *
 *   NativeBackendRegistry.cpp   — lazy provider singletons + registry
 *   NativeBackendValueInit.cpp  — initialize_tensor_evalue / initialize_values
 *   NativeBackendBuffers.cpp    — materialize_buffers / upload_constants
 *   NativeBackendExecute.cpp    — bind_inputs / bind_outputs / execute_step
 */

#include <executorch/backends/native/NativeBackendInternal.h>

#include <executorch/backends/native/routers/GreedyRouter.h>

#include <executorch/schema/program_generated.h>

#include <algorithm>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

class NativeBackend final : public ::executorch::runtime::BackendInterface {
 public:
  ~NativeBackend() override = default;

  bool is_available() const override {
    return true;
  }

  Result<DelegateHandle*> init(
      BackendInitContext& ctx,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> /*compile_specs*/) const override {
    if (!processed || !processed->data() || processed->size() == 0) {
      return Error::InvalidArgument;
    }

    auto* program = executorch_flatbuffer::GetProgram(processed->data());
    if (!program)
      return Error::InvalidProgram;

    auto plans = program->execution_plan();
    if (!plans || plans->size() == 0)
      return Error::InvalidProgram;

    auto* runtime_alloc = ctx.get_runtime_allocator();
    if (!runtime_alloc)
      return Error::InvalidState;
    void* mem = runtime_alloc->allocate(
        sizeof(DelegateInstance), alignof(DelegateInstance));
    if (!mem)
      return Error::MemoryAllocationFailed;
    auto* d = new (mem) DelegateInstance();

    d->graph = std::make_unique<Graph>(plans->Get(0), program);

    // Borrow the process-wide registry (constructed lazily on first
    // call). DelegateInstance does not own Runtimes or RuntimeContexts;
    // it only owns the per-program Engines below.
    auto& registry = native_runtime_registry();

    // Read load-time backend option "compute_unit" if provided.
    // Value is a `|`-separated list of compute units to enable:
    //   - unset / "auto"    : enable cpu + metal (the standard set)
    //   - "cpu"             : cpu only (host pool always retained)
    //   - "metal"           : metal only (requires ET_NATIVE_HAS_METAL
    //                         compile flag; otherwise no compute provider)
    //   - "cpu|metal"       : both, explicit form
    //   - "fake_accel"      : routing-test scaffolding (CpuRuntime
    //                         restricted to add/mul); rarely used
    // Unknown unit names are ignored (logged) so unrecognized values
    // don't silently broaden the set; if no recognized units result,
    // we fall back to "auto" so the load doesn't produce an empty plan.
    //
    // The host pool is always retained — it isn't a compute unit, it
    // serves graph IO and host-pool allocations regardless of target.
    const char* compute_unit_filter = nullptr;
    {
      auto r = ctx.get_runtime_spec<const char*>("compute_unit");
      if (r.ok()) {
        compute_unit_filter = r.get();
      }
    }
    // Parse the `|`-separated allowlist into a small set of unit names.
    // "auto" (or empty) means "use the standard set" (cpu + metal).
    // C++ stdlib has no string split; std::getline with a stringstream
    // and a custom delimiter is the idiomatic stdlib idiom.
    std::unordered_set<std::string> allowed_units;
    bool accept_all = (compute_unit_filter == nullptr);
    if (compute_unit_filter) {
      std::stringstream ss(compute_unit_filter);
      std::string tok;
      while (std::getline(ss, tok, '|')) {
        if (tok == "auto") {
          accept_all = true;
        } else if (!tok.empty()) {
          allowed_units.insert(std::move(tok));
        }
      }
      if (allowed_units.empty() && !accept_all) {
        ET_LOG(
            Info,
            "[options] compute_unit='%s': no recognized units; defaulting to auto",
            compute_unit_filter);
        accept_all = true;
      }
    }
    auto unit_allowed = [&](const char* name) -> bool {
      if (accept_all)
        return true;
      return allowed_units.count(name) > 0;
    };

    auto avail = registry.available();
    // Build the candidate-providers list:
    //   1. Host pool (always; from the eager registry).
    //   2. Optional fake_accel (only if explicitly requested via
    //      compute_unit).
    //   3. Metal (lazy; only if compile flag set AND compute_unit
    //      allows it AND device init succeeds).
    //   4. CPU (lazy; the universal fallback; only if compute_unit
    //      allows it).
    //
    // Ordering matters: the router prefers earlier providers when
    // multiple can_run an op. So we put more specialized providers
    // (fake_accel, Metal) ahead of CPU. CPU is always last so it can
    // serve as the catch-all fallback for any op the others reject.
    std::vector<Runtime*> candidate_providers(avail.begin(), avail.end());
    // 2. fake_accel (opt-in only — never enabled by "auto"; routing
    //    tests must request it explicitly).
    if (allowed_units.count("fake_accel") > 0) {
      candidate_providers.push_back(lazy_fake_accel_runtime());
    }
#ifdef ET_NATIVE_HAS_METAL
    // 3. Metal.
    if (unit_allowed("metal")) {
      if (auto* m = lazy_metal_runtime()) {
        candidate_providers.push_back(m);
      }
    }
#endif
    // 4. CPU last (always-available fallback when allowed).
    if (unit_allowed("cpu")) {
      candidate_providers.push_back(lazy_cpu_runtime());
    }

    d->owned_instances.reserve(candidate_providers.size());
    std::vector<Engine*> raw_instances;
    raw_instances.reserve(candidate_providers.size());
    std::vector<Runtime*> raw_providers;
    raw_providers.reserve(candidate_providers.size());
    for (auto* p : candidate_providers) {
      auto inst = p->instantiate();
      raw_instances.push_back(inst.get());
      raw_providers.push_back(p);
      d->owned_instances.push_back(std::move(inst));
    }
    if (compute_unit_filter) {
      ET_LOG(
          Info,
          "[options] compute_unit='%s': %zu provider(s) active",
          compute_unit_filter,
          raw_providers.size());
    }

    GreedyRouter router;
    RouterOptions opts;
    auto plan_result = router.route(
        *d->graph,
        Span<Runtime* const>(raw_providers.data(), raw_providers.size()),
        Span<Engine* const>(raw_instances.data(), raw_instances.size()),
        ctx.get_named_data_map(),
        opts);
    if (!plan_result.ok()) {
      d->~DelegateInstance();
      return plan_result.error();
    }
    d->plan = std::move(plan_result.get());

    // Post-route summary: surface the partition decision so it's
    // visible without enabling Debug logging. Each step is one of:
    //   ComputeStep    — segment of N kernel instructions on a runtime
    //   TransferStep   — host<->device byte movement
    //   JumpFalseStep  — host-side conditional jump
    //   MoveStep       — host-side EValue assignment
    {
      ET_LOG(
          Info,
          "[router] partition: %zu providers, %zu steps",
          d->plan.providers.size(),
          d->plan.steps.size());
      auto provider_name = [&](RuntimeIndex p) -> std::string {
        if (p < d->plan.providers.size() && d->plan.providers[p]) {
          return std::string(d->plan.providers[p]->name());
        }
        return "?";
      };
      for (size_t i = 0; i < d->plan.providers.size(); ++i) {
        std::string name = provider_name(static_cast<RuntimeIndex>(i));
        ET_LOG(Info, "[router]   provider[%zu]: %s", i, name.c_str());
      }
      for (size_t i = 0; i < d->plan.steps.size(); ++i) {
        std::visit(
            [&](auto&& s) {
              using T = std::decay_t<decltype(s)>;
              if constexpr (std::is_same_v<T, ComputeStep>) {
                std::string name = provider_name(s.runtime_idx);
                ET_LOG(
                    Info,
                    "[router]   step[%zu]: ComputeStep on provider[%u] (%s), source_pc=%u",
                    i,
                    s.runtime_idx,
                    name.c_str(),
                    s.source_pc);
              } else if constexpr (std::is_same_v<T, TransferStep>) {
                std::string sn = provider_name(s.src_idx);
                std::string dn = provider_name(s.dst_idx);
                ET_LOG(
                    Info,
                    "[router]   step[%zu]: TransferStep src=provider[%u](%s) vid=%u -> dst=provider[%u](%s) vid=%u",
                    i,
                    s.src_idx,
                    sn.c_str(),
                    s.src_value_id,
                    s.dst_idx,
                    dn.c_str(),
                    s.dst_value_id);
              } else if constexpr (std::is_same_v<T, JumpFalseStep>) {
                ET_LOG(
                    Info,
                    "[router]   step[%zu]: JumpFalseStep pred_vid=%u dst_step=%zu",
                    i,
                    s.pred_value_id,
                    s.dst_step_idx);
              } else if constexpr (std::is_same_v<T, MoveStep>) {
                ET_LOG(
                    Info,
                    "[router]   step[%zu]: MoveStep vid %u -> %u",
                    i,
                    s.src_value_id,
                    s.dst_value_id);
              }
            },
            d->plan.steps[i]);
      }
    }

    size_t num_orig = d->graph->num_values();
    size_t total_size = num_orig;
    // Engines materialize TensorImpls for any router-minted mirror
    // value_ids in their allocate_buffers; we just need the central
    // EValue array sized to fit them. Walk every alloc_plan to find
    // the highest value_id any engine will touch.
    for (const auto& plan : d->plan.alloc_plans) {
      for (const auto& req : plan) {
        total_size = std::max<size_t>(total_size, req.value_id + 1);
      }
    }
    d->values.reserve(total_size);
    d->values.resize(total_size);
    if (auto e = initialize_values(d, runtime_alloc); e != Error::Ok) {
      d->~DelegateInstance();
      return e;
    }
    if (auto e = materialize_buffers(d); e != Error::Ok) {
      d->~DelegateInstance();
      return e;
    }
    if (auto e = upload_constants(d, ctx.get_named_data_map());
        e != Error::Ok) {
      d->~DelegateInstance();
      return e;
    }

    // Inform every engine of the graph IO bindings so each can build
    // its internal IO table for per-execute bind_inputs / bind_outputs.
    for (Engine* inst : d->plan.instances) {
      if (!inst)
        continue;
      auto err = inst->set_io_bindings(
          Span<const InputBinding>(
              d->plan.inputs.data(), d->plan.inputs.size()),
          Span<const OutputBinding>(
              d->plan.outputs.data(), d->plan.outputs.size()));
      if (err != Error::Ok) {
        d->~DelegateInstance();
        return err;
      }
    }

    ET_LOG(
        Info,
        "NativeBackend: initialized with %zu steps",
        d->plan.steps.size());
    return reinterpret_cast<DelegateHandle*>(d);
  }

  Error execute(
      BackendExecutionContext& /*ctx*/,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    auto* d = static_cast<DelegateInstance*>(handle);
    if (!d)
      return Error::InvalidState;

    if (d->poisoned)
      return Error::Internal;

    if (auto e = bind_inputs(d, args); e != Error::Ok)
      return e;
    if (auto e = bind_outputs(d, args); e != Error::Ok)
      return e;

    Error first_err = Error::Ok;
    std::vector<EventId> observed_signals;
    observed_signals.reserve(d->plan.steps.size());

    size_t pc = 0;
    size_t hops = 0;
    while (pc < d->plan.steps.size()) {
      if (++hops > d->plan.max_hops) {
        ET_LOG(
            Error, "NativeBackend: max_hops exceeded — malformed back edge?");
        first_err = Error::InvalidProgram;
        break;
      }

      const Step& step = d->plan.steps[pc];

      if (auto* jf = std::get_if<JumpFalseStep>(&step)) {
        if (jf->dst_step_idx == kUnresolvedStep) {
          ET_LOG(
              Error,
              "NativeBackend: unresolved JumpFalseStep dst at pc=%zu",
              pc);
          first_err = Error::InvalidProgram;
          break;
        }
        for (EventId id : jf->wait_for) {
          if (id >= d->plan.events.size())
            continue;
          auto& slot = d->plan.events[id];
          if (slot.event && slot.owner) {
            Error e = slot.owner->wait(slot.event.get());
            if (e != Error::Ok && first_err == Error::Ok)
              first_err = e;
          }
        }
        if (first_err != Error::Ok)
          break;

        if (jf->pred_value_id >= d->values.size()) {
          first_err = Error::InvalidState;
          break;
        }
        auto pred_r = parse_cond_value(d->values[jf->pred_value_id]);
        if (!pred_r.ok()) {
          first_err = pred_r.error();
          break;
        }
        ET_LOG(
            Debug,
            "[cf] JumpFalseStep pc=%zu pred_vid=%u pred=%d wait_for=%zu next_pc=%zu",
            pc,
            jf->pred_value_id,
            static_cast<int>(pred_r.get()),
            jf->wait_for.size(),
            pred_r.get() ? (pc + 1) : jf->dst_step_idx);
        pc = pred_r.get() ? (pc + 1) : jf->dst_step_idx;
        continue;
      }

      Error e = execute_step(d, step);
      if (e != Error::Ok) {
        first_err = e;
        break;
      }
      EventId sig = kNoEvent;
      if (auto* cs = std::get_if<ComputeStep>(&step))
        sig = cs->signal;
      else if (auto* ts = std::get_if<TransferStep>(&step))
        sig = ts->signal;
      if (sig != kNoEvent)
        observed_signals.push_back(sig);

      ++pc;
    }

    std::unordered_set<EventId> observed_set(
        observed_signals.begin(), observed_signals.end());
    for (EventId id : d->plan.terminal_events) {
      if (id >= d->plan.events.size())
        continue;
      if (observed_set.count(id) == 0)
        continue;
      auto& slot = d->plan.events[id];
      if (slot.event && slot.owner) {
        Error e = slot.owner->wait(slot.event.get());
        if (e != Error::Ok && first_err == Error::Ok)
          first_err = e;
      }
    }

    if (first_err != Error::Ok) {
      d->poisoned = true;
      return first_err;
    }

    // Per-output post-execute pass:
    //   1. Non-tensor outputs (scalars, lists): copy the EValue to the
    //      caller's slot.
    //   2. Tensor outputs: propagate computed shape onto the caller's
    //      tensor (caller may have passed a stale shape; the kernel
    //      may have produced a different one).
    //
    // No memcpy fallback: by protocol, bind_outputs aliased the host
    // buffer to the caller's pointer, and downloads (if any) wrote
    // through that alias, so values[vid].data_ptr == caller_t.data_ptr
    // always at this point. A mismatch is a bind/transfer bug; we log
    // and report Error::Internal rather than silently memcpy from
    // possibly-stale storage.
    size_t n_in = d->plan.inputs.size();
    size_t n_out = d->plan.outputs.size();
    for (size_t i = 0; i < n_out && (n_in + i) < args.size(); ++i) {
      const auto& ob = d->plan.outputs[i];
      uint32_t vid = ob.value_id;
      if (vid >= d->values.size())
        continue;
      EValue* arg_ev = args[n_in + i];
      if (!arg_ev)
        continue;
      if (!d->values[vid].isTensor()) {
        *arg_ev = d->values[vid];
        continue;
      }
      if (arg_ev->isTensor()) {
        auto& host_t = d->values[vid].toTensor();
        auto& caller_t = arg_ev->toTensor();
        (void)::executorch::runtime::resize_tensor(caller_t, host_t.sizes());
        const void* sp = host_t.const_data_ptr();
        void* dp = caller_t.mutable_data_ptr();
        if (sp && dp && sp != dp) {
          ET_LOG(
              Error,
              "NativeBackend: output vid=%u: bind/transfer alias broken "
              "(host_ptr=%p != caller_ptr=%p). Engine wrote to internal "
              "storage instead of caller's buffer.",
              vid,
              sp,
              dp);
          d->poisoned = true;
          return Error::Internal;
        }
      }
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (!handle)
      return;
    auto* d = static_cast<DelegateInstance*>(handle);
    d->~DelegateInstance();
  }
};

namespace {
NativeBackend g_backend_v2;
Backend g_backend_record{"NativeBackend", &g_backend_v2};
auto g_register = ::executorch::runtime::register_backend(g_backend_record);
} // namespace

} // namespace native
} // namespace backends
} // namespace executorch

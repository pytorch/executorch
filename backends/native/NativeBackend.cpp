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
 */

#include <executorch/backends/native/ir/Plan.h>
#include <executorch/backends/native/core/RuntimeRegistry.h>
#include <executorch/backends/native/core/Router.h>
#include <executorch/backends/native/ir/Step.h>
#include <executorch/backends/native/runtimes/cpu/CpuRuntime.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntime.h>
#include <executorch/backends/native/routers/GreedyRouter.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#ifdef NATIVE_HAS_METAL
#include <executorch/backends/native/runtimes/metal/MetalRuntime.h>
#endif

#include <executorch/backends/native/ir/GraphTypes.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

#include <executorch/schema/program_generated.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

namespace {

using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::Result;
using ValueType = ::executorch::backends::portable::ValueType;
using ::executorch::runtime::Span;

using ::executorch::aten::DimOrderType;
using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::aten::StridesType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;

using ::executorch::backends::portable::Graph;

constexpr RuntimeIndex kInvalidRuntimeIdx =
    std::numeric_limits<RuntimeIndex>::max();

/**
 * Per-program state held across init/execute/destroy. One per loaded
 * delegate. See DelegateInstance in §6 of the design doc.
 *
 * Note: the RuntimeRegistry (and the Runtimes / RuntimeContexts it
 * owns) lives at process scope, not on this struct — see
 * native_runtime_registry() below. Only the per-program Engines
 * (`owned_instances`) and routing state (`graph`, `plan`, `values`,
 * etc.) are owned here.
 */
struct DelegateInstance {
  // Per-program Engines. One per available process-wide Runtime,
  // produced by Runtime::instantiate() at init. Engines hold a
  // non-owning reference to their Runtime's process-wide
  // RuntimeContext.
  std::vector<std::unique_ptr<Engine>> owned_instances;

  // Parsed program — wrapped behind Graph. Downstream code reaches the
  // serialized program ONLY through `graph`; nothing else holds a raw
  // flatbuffer pointer. See §3 of PORTABLE_BACKEND_API_PROPOSAL.md.
  std::unique_ptr<Graph> graph;

  // Universal value array, indexed by value_id. Holds EValues for
  // everything (scalars, lists, tensors). For tensor EValues, the
  // TensorImpl::data_ptr is updated by engines (during allocate_buffers
  // and bind_inputs/bind_outputs) for host-addressable buffers, and
  // refreshed at dispatch time per op-arg from each engine's internal
  // value_to_buffer_ table.
  std::vector<EValue> values;

  // Frozen routing decision.
  Plan plan;

  // TensorImpl storage for tensor EValues we materialize from the
  // flatbuffer (sizes / dim_order / strides arrays + the TensorImpl
  // structs themselves). RAII-cleaned via vector<unique_ptr>.
  struct TensorMeta {
    std::unique_ptr<SizesType[]> sizes;
    std::unique_ptr<DimOrderType[]> dim_order;
    std::unique_ptr<StridesType[]> strides;
    std::unique_ptr<TensorImpl> impl;
  };
  std::vector<TensorMeta> tensor_metas;

  bool poisoned = false;

  // value_id → which Engine claimed this vid during materialize_buffers
  // (or kInvalidRuntimeIdx if the vid has no AllocRequest, e.g. a
  // scalar EValue or a constant tracked engine-internally). Sized to
  // d->values.size(). The single piece of cross-engine bookkeeping
  // NativeBackend retains.
  std::vector<RuntimeIndex> value_owner;

  // graph_input_or_output_vid → list of (mirror_id, owning_engine_idx)
  // for each device-side mirror of that vid. Built once at end of
  // init() by walking plan.mirror_values + value_owner. Used by
  // bind_inputs / bind_outputs to bucket per-engine IO work in O(1).
  std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, RuntimeIndex>>>
      mirror_index;

  ~DelegateInstance() {
    // Drain (no-op for CPU; matters when GPU instances are present).
    for (auto* inst : plan.instances) {
      if (inst)
        inst->drain();
    }
    // Buffers are owned by their engines; engine destructors release
    // them. NativeBackend holds no Buffer*s of its own.
  }
};

// Build the default ProviderSet. Slot 0 is reserved for HostPoolRuntime —
// the canonical home for boundary values (graph IO, cross-runtime
// intermediates). HostPool is allocator-only (can_run() == false), so
// the router never assigns ops to it. Compute providers occupy slots 1+.
std::vector<std::unique_ptr<Runtime>> make_default_runtimes() {
  std::vector<std::unique_ptr<Runtime>> ps;
  ps.push_back(std::make_unique<HostPoolRuntime>());

  const char* fake = std::getenv("NATIVE_USE_FAKE_ACCEL");
  if (fake && fake[0] == '1') {
    ET_LOG(
        Info,
        "NativeBackend: registering CpuRuntime(\"fake_accel\") for routing tests");
    std::unordered_set<std::string> allow = {
        "aten::add",
        "aten::add.Tensor",
        "aten::mul",
        "aten::mul.Tensor",
    };
    ps.push_back(std::make_unique<CpuRuntime>("fake_accel", std::move(allow)));
    // Real CPU appended below as fallback.
  }

#ifdef NATIVE_HAS_METAL
  if (!fake || fake[0] != '1') {
    const char* disable_metal = std::getenv("NATIVE_DISABLE_METAL");
    if (!disable_metal || disable_metal[0] != '1') {
      auto metal = std::make_unique<MetalRuntime>();
      if (metal->stream_ready()) {
        ET_LOG(Info, "NativeBackend: registering MetalRuntime");
        ps.push_back(std::move(metal));
      } else {
        ET_LOG(
            Info,
            "NativeBackend: MetalRuntime unavailable; CPU-only mode");
      }
    }
  }
#endif

  // Real CPU always last so it serves as fallback for ops no specialised
  // non-host provider accepted.
  ps.push_back(std::make_unique<CpuRuntime>());
  return ps;
}

// Process-wide RuntimeRegistry. Constructed once on first call (Meyers
// singleton; thread-safe in C++11+); lives until process exit.
//
// The Runtimes it owns — and the RuntimeContexts they own (kernel
// caches, JIT artifacts, GPU command queues, etc., once non-trivial
// runtimes land) — are therefore shared across every DelegateInstance
// in the process. Per-program state still lives on each
// DelegateInstance's `owned_instances` (the Engines).
//
// Contract for non-trivial RuntimeContext state added later: must be
// safe for concurrent reads + occasional writes from multiple
// delegate threads (see threading model in design doc).
RuntimeRegistry& native_runtime_registry() {
  static RuntimeRegistry registry(make_default_runtimes());
  return registry;
}

// Initialize one tensor EValue from the flatbuffer Tensor metadata.
Error initialize_tensor_evalue(DelegateInstance* d, uint32_t value_id) {
  const auto& graph = *d->graph;
  auto sizes = graph.tensor_sizes(value_id);
  auto dim_order_in = graph.tensor_dim_order(value_id);
  size_t dim = sizes.size();

  DelegateInstance::TensorMeta meta;
  meta.sizes.reset(new SizesType[dim]);
  meta.dim_order.reset(new DimOrderType[dim]);
  meta.strides.reset(new StridesType[dim]);

  for (size_t i = 0; i < dim; ++i) {
    meta.sizes[i] = sizes[i];
  }
  if (dim_order_in.size() == dim) {
    for (size_t i = 0; i < dim; ++i) {
      meta.dim_order[i] = dim_order_in[i];
    }
  } else {
    for (size_t i = 0; i < dim; ++i) {
      meta.dim_order[i] = static_cast<DimOrderType>(i);
    }
  }
  auto status = ::executorch::runtime::dim_order_to_stride(
      meta.sizes.get(), meta.dim_order.get(), dim, meta.strides.get());
  if (status != Error::Ok)
    return status;

  ScalarType dtype = graph.tensor_dtype(value_id);
  meta.impl.reset(new TensorImpl(
      dtype,
      static_cast<ssize_t>(dim),
      meta.sizes.get(),
      /*data=*/nullptr,
      meta.dim_order.get(),
      meta.strides.get(),
      ::executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND));

  d->values[value_id] = EValue(Tensor(meta.impl.get()));
  d->tensor_metas.push_back(std::move(meta));
  return Error::Ok;
}

Error initialize_values(
    DelegateInstance* d,
    ::executorch::runtime::MemoryAllocator* runtime_alloc) {
  using ::executorch::runtime::BoxedEvalueList;

  size_t n = d->graph->num_values();
  if (n == 0)
    return Error::Ok;

  if (d->values.size() < n) {
    return Error::InvalidState;
  }

  for (uint32_t i = 0; i < n; ++i) {
    switch (d->graph->value_type(i)) {
      case ValueType::None:
        d->values[i] = EValue();
        break;
      case ValueType::Int:
        d->values[i] = EValue(d->graph->int_value(i));
        break;
      case ValueType::Double:
        d->values[i] = EValue(d->graph->double_value(i));
        break;
      case ValueType::Bool:
        d->values[i] = EValue(d->graph->bool_value(i));
        break;
      case ValueType::Tensor:
        if (auto e = initialize_tensor_evalue(d, i); e != Error::Ok) {
          return e;
        }
        break;
      case ValueType::IntList: {
        auto items = d->graph->int_list_member_ids(i);
        size_t cnt = items.size();
        EValue** evalp_list = runtime_alloc->allocateList<EValue*>(cnt);
        int64_t* int_list = runtime_alloc->allocateList<int64_t>(cnt);
        if (!evalp_list || !int_list) {
          return Error::MemoryAllocationFailed;
        }
        for (size_t j = 0; j < cnt; ++j) {
          int64_t vidx = items[j];
          if (vidx < 0 || static_cast<size_t>(vidx) >= n) {
            return Error::InvalidProgram;
          }
          evalp_list[j] = &d->values[static_cast<size_t>(vidx)];
        }
        auto* boxed_mem =
            runtime_alloc->allocateInstance<BoxedEvalueList<int64_t>>();
        if (!boxed_mem)
          return Error::MemoryAllocationFailed;
        auto* boxed =
            new (boxed_mem) BoxedEvalueList<int64_t>(evalp_list, int_list, cnt);
        d->values[i] = EValue(boxed);
      } break;
      case ValueType::TensorList:
        d->values[i] = EValue();
        break;
      case ValueType::OptionalTensorList: {
        auto items = d->graph->tensor_list_member_ids(i);
        size_t cnt = items.size();
        EValue** evalp_list = runtime_alloc->allocateList<EValue*>(cnt);
        std::optional<::executorch::aten::Tensor>* opt_list =
            runtime_alloc->allocateList<
                std::optional<::executorch::aten::Tensor>>(cnt);
        if (!evalp_list || !opt_list) {
          return Error::MemoryAllocationFailed;
        }
        EValue* none_ev = nullptr;
        for (size_t j = 0; j < cnt; ++j) {
          int32_t vidx = items[j];
          if (vidx == -1) {
            if (!none_ev) {
              void* mem = runtime_alloc->allocateInstance<EValue>();
              if (!mem) return Error::MemoryAllocationFailed;
              none_ev = new (mem) EValue();
            }
            evalp_list[j] = none_ev;
          } else if (vidx < 0 || static_cast<size_t>(vidx) >= n) {
            return Error::InvalidProgram;
          } else {
            evalp_list[j] = &d->values[static_cast<size_t>(vidx)];
          }
          new (&opt_list[j]) std::optional<::executorch::aten::Tensor>();
        }
        auto* boxed_mem = runtime_alloc->allocateInstance<
            BoxedEvalueList<std::optional<::executorch::aten::Tensor>>>();
        if (!boxed_mem)
          return Error::MemoryAllocationFailed;
        auto* boxed = new (boxed_mem)
            BoxedEvalueList<std::optional<::executorch::aten::Tensor>>(
                evalp_list, opt_list, cnt);
        d->values[i] = EValue(boxed);
      } break;
      case ValueType::Other:
        d->values[i] = EValue();
        break;
    }
  }
  return Error::Ok;
}

// Materialize EValues for router-minted mirror value_ids. Each mirror
// inherits dtype/shape/strides from its host-side value_id. data_ptr
// stays null; engines populate it when they claim the mirror_id at
// allocate_buffers time (host-addressable case) or leave it null
// (discrete-GPU case).
Error materialize_mirror_values(DelegateInstance* d) {
  if (d->plan.mirror_values.empty())
    return Error::Ok;

  for (const auto& mv : d->plan.mirror_values) {
    if (mv.mirror_id >= d->values.size()) {
      ET_LOG(
          Error,
          "materialize_mirror_values: mirror_id=%u >= values.size()=%zu",
          mv.mirror_id,
          d->values.size());
      return Error::InvalidState;
    }
    if (mv.source_value_id >= d->values.size() ||
        !d->values[mv.source_value_id].isTensor()) {
      ET_LOG(
          Error,
          "materialize_mirror_values: source value_id=%u is not a tensor",
          mv.source_value_id);
      return Error::InvalidProgram;
    }

    auto& src = d->values[mv.source_value_id].toTensor();
    auto* src_impl = src.unsafeGetTensorImpl();
    size_t dim = src.dim();

    DelegateInstance::TensorMeta tm;
    tm.sizes.reset(new SizesType[dim]);
    tm.dim_order.reset(new DimOrderType[dim]);
    tm.strides.reset(new StridesType[dim]);
    for (size_t i = 0; i < dim; ++i) {
      tm.sizes[i] = src.size(i);
      tm.dim_order[i] = src_impl->dim_order()[i];
      tm.strides[i] = src.strides()[i];
    }
    tm.impl.reset(new TensorImpl(
        src.scalar_type(),
        static_cast<ssize_t>(dim),
        tm.sizes.get(),
        /*data=*/nullptr,
        tm.dim_order.get(),
        tm.strides.get(),
        ::executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND));

    d->values[mv.mirror_id] = EValue(Tensor(tm.impl.get()));
    d->tensor_metas.push_back(std::move(tm));
  }
  return Error::Ok;
}

// Bid-auction allocation pass. Walks each provider's alloc_plan in
// provider index order (HostPool slot 0 first, then non-host engines).
// Each engine claims requests it wants; declined requests fall through
// to subsequent engines (HostPool is the floor for HostMirror/HostOnly).
//
// Engines internally:
//   - Allocate a Buffer for each claimed vid.
//   - Insert (vid → Buffer*) into their own value_to_buffer_ table.
//   - For host-addressable Buffers, set data_ptr on values[vid].
//
// NativeBackend records value_owner[vid] = runtime_idx for each Claimed
// request and asserts no AllocRequest's vid is left unowned.
Error materialize_buffers(DelegateInstance* d) {
  d->value_owner.assign(d->values.size(), kInvalidRuntimeIdx);

  for (size_t p = 0; p < d->plan.alloc_plans.size(); ++p) {
    auto& reqs = d->plan.alloc_plans[p];
    if (reqs.empty())
      continue;
    if (p >= d->plan.instances.size() || !d->plan.instances[p]) {
      return Error::InvalidState;
    }
    Engine* inst = d->plan.instances[p];

    std::vector<Engine::AllocClaim> claims(reqs.size());
    auto err = inst->allocate_buffers(
        Span<const Engine::AllocRequest>(reqs.data(), reqs.size()),
        Span<EValue>(d->values.data(), d->values.size()),
        Span<Engine::AllocClaim>(claims.data(), claims.size()));
    if (err != Error::Ok)
      return err;

    for (size_t i = 0; i < claims.size(); ++i) {
      if (claims[i] != Engine::AllocClaim::Claimed) continue;
      uint32_t vid = reqs[i].value_id;
      if (vid >= d->value_owner.size()) {
        ET_LOG(Error,
               "materialize_buffers: vid=%u out of range (value_owner.size=%zu)",
               vid, d->value_owner.size());
        return Error::InvalidState;
      }
      // First-claim-wins: a UMA collapse engine may claim a HostMirror
      // request that HostPool would have claimed in the next pass; we
      // record the FIRST claimer. If a subsequent engine also claims
      // (which shouldn't happen by protocol), we keep the first.
      if (d->value_owner[vid] == kInvalidRuntimeIdx) {
        d->value_owner[vid] = static_cast<RuntimeIndex>(p);
      }
    }
  }

  // Assert no AllocRequest's vid was left unowned.
  for (size_t p = 0; p < d->plan.alloc_plans.size(); ++p) {
    for (const auto& req : d->plan.alloc_plans[p]) {
      if (req.value_id >= d->value_owner.size() ||
          d->value_owner[req.value_id] == kInvalidRuntimeIdx) {
        ET_LOG(Error,
               "materialize_buffers: vid=%u (kind=%s) was not claimed by any "
               "engine — router/init bug",
               req.value_id, to_string(req.kind));
        return Error::InvalidState;
      }
    }
  }

  ET_LOG(Debug, "[mem] materialize_buffers: bid auction complete");
  return Error::Ok;
}

// Constants pass: drive each engine's upload_constants from the
// per-provider const_plans the router emitted. Symmetric to
// materialize_buffers / alloc_plans: route() is pure planning, all
// engine I/O happens here.
//
// Each engine independently materializes its constants (zero-copy NDM
// alias on CPU / Apple-Silicon Metal; device-side load on discrete
// GPU). Engines track lifetime and value→Buffer mapping internally;
// nothing leaves through this API.
Error upload_constants(DelegateInstance* d, const ::executorch::runtime::NamedDataMap* ndm) {
  if (d->plan.const_plans.empty()) return Error::Ok;
  // If any provider has constants to upload, ndm must be present.
  for (const auto& reqs : d->plan.const_plans) {
    if (!reqs.empty() && !ndm) {
      ET_LOG(
          Error,
          "upload_constants: const_plans non-empty but NamedDataMap is null");
      return Error::InvalidArgument;
    }
  }
  for (size_t p = 0; p < d->plan.const_plans.size(); ++p) {
    const auto& reqs = d->plan.const_plans[p];
    if (reqs.empty()) continue;
    if (p >= d->plan.instances.size() || !d->plan.instances[p]) {
      return Error::InvalidState;
    }
    Engine* inst = d->plan.instances[p];
    auto err = inst->upload_constants(
        *ndm,
        Span<const Engine::ConstRequest>(reqs.data(), reqs.size()));
    if (err != Error::Ok) {
      ET_LOG(
          Error,
          "upload_constants: provider %zu (%s) failed (%zu requests)",
          p,
          d->plan.providers[p] ? std::string(d->plan.providers[p]->name()).c_str() : "?",
          reqs.size());
      return err;
    }
    for (size_t i = 0; i < reqs.size(); ++i) {
      const auto& req = reqs[i];
      ET_LOG(
          Debug,
          "[mem] upload_constants[%zu] value_id=%u key='%.*s' provider=%zu (%s)",
          i,
          req.value_id,
          static_cast<int>(req.ndm_key.size()),
          req.ndm_key.data(),
          p,
          d->plan.providers[p] ? std::string(d->plan.providers[p]->name()).c_str() : "?");
    }
  }
  return Error::Ok;
}

// Per-execute IO binding: bucket caller args per owning engine
// (using value_owner + mirror_index), then batch-call each engine's
// bind_inputs.
Error bind_inputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t n_engines = d->plan.instances.size();

  // Per-engine buckets.
  std::vector<std::vector<EValue>> caller_buckets(n_engines);
  std::vector<std::vector<uint32_t>> vid_buckets(n_engines);

  for (size_t i = 0; i < n_in && i < args.size(); ++i) {
    const auto& ib = d->plan.inputs[i];
    uint32_t vid = ib.value_id;
    if (vid >= d->values.size()) return Error::InvalidArgument;
    if (!args[i]->isTensor() || !d->values[vid].isTensor())
      continue;
    void* caller_ptr = args[i]->toTensor().mutable_data_ptr();
    if (!caller_ptr) continue;

    // Host-side bucket (the engine that claimed vid in materialize_buffers).
    RuntimeIndex host_owner = (vid < d->value_owner.size())
        ? d->value_owner[vid] : kInvalidRuntimeIdx;
    if (host_owner == kInvalidRuntimeIdx) {
      // No AllocRequest for this vid (router didn't emit one). Default
      // to HostPool — IO must live somewhere.
      host_owner = kHostIdx;
    }
    if (host_owner < n_engines) {
      caller_buckets[host_owner].push_back(*args[i]);
      vid_buckets[host_owner].push_back(vid);
    }

    // Mirror-side buckets.
    auto mit = d->mirror_index.find(vid);
    if (mit != d->mirror_index.end()) {
      for (const auto& [mirror_id, eng_idx] : mit->second) {
        if (eng_idx >= n_engines) continue;
        caller_buckets[eng_idx].push_back(*args[i]);
        vid_buckets[eng_idx].push_back(mirror_id);
      }
    }
  }

  // Dispatch per engine.
  for (size_t e = 0; e < n_engines; ++e) {
    if (vid_buckets[e].empty()) continue;
    Engine* inst = d->plan.instances[e];
    if (!inst) continue;
    auto err = inst->bind_inputs(
        Span<EValue>(d->values.data(), d->values.size()),
        Span<const EValue>(
            caller_buckets[e].data(), caller_buckets[e].size()),
        Span<const uint32_t>(
            vid_buckets[e].data(), vid_buckets[e].size()));
    if (err != Error::Ok) return err;
  }
  return Error::Ok;
}

Error bind_outputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t n_out = d->plan.outputs.size();
  size_t n_engines = d->plan.instances.size();

  std::vector<std::vector<EValue>> caller_buckets(n_engines);
  std::vector<std::vector<uint32_t>> vid_buckets(n_engines);

  for (size_t i = 0; i < n_out && (n_in + i) < args.size(); ++i) {
    const auto& ob = d->plan.outputs[i];
    uint32_t vid = ob.value_id;
    if (vid >= d->values.size()) return Error::InvalidArgument;

    EValue* arg_ev = args[n_in + i];
    if (!arg_ev || !arg_ev->isTensor() || !d->values[vid].isTensor())
      continue;
    void* caller_ptr = arg_ev->toTensor().mutable_data_ptr();
    if (!caller_ptr) continue;

    RuntimeIndex host_owner = (vid < d->value_owner.size())
        ? d->value_owner[vid] : kInvalidRuntimeIdx;
    if (host_owner == kInvalidRuntimeIdx) {
      host_owner = kHostIdx;
    }
    if (host_owner < n_engines) {
      caller_buckets[host_owner].push_back(*arg_ev);
      vid_buckets[host_owner].push_back(vid);
    }

    auto mit = d->mirror_index.find(vid);
    if (mit != d->mirror_index.end()) {
      for (const auto& [mirror_id, eng_idx] : mit->second) {
        if (eng_idx >= n_engines) continue;
        caller_buckets[eng_idx].push_back(*arg_ev);
        vid_buckets[eng_idx].push_back(mirror_id);
      }
    }
  }

  for (size_t e = 0; e < n_engines; ++e) {
    if (vid_buckets[e].empty()) continue;
    Engine* inst = d->plan.instances[e];
    if (!inst) continue;
    auto err = inst->bind_outputs(
        Span<EValue>(d->values.data(), d->values.size()),
        Span<const EValue>(
            caller_buckets[e].data(), caller_buckets[e].size()),
        Span<const uint32_t>(
            vid_buckets[e].data(), vid_buckets[e].size()));
    if (err != Error::Ok) return err;
  }
  return Error::Ok;
}

inline Event* resolve_event(DelegateInstance* d, EventId id) {
  if (id == kNoEvent || id >= d->plan.events.size()) return nullptr;
  return d->plan.events[id].event.get();
}

::executorch::runtime::Result<bool> parse_cond_value(const EValue& cond) {
  if (cond.isTensor()) {
    const auto& t = cond.toTensor();
    if (t.scalar_type() != ScalarType::Bool) {
      ET_LOG(
          Error,
          "parse_cond_value: expected Bool tensor, got dtype %d",
          static_cast<int>(t.scalar_type()));
      return Error::InvalidProgram;
    }
    const bool* data = t.const_data_ptr<bool>();
    if (!data) {
      ET_LOG(Error, "parse_cond_value: predicate tensor has null data_ptr");
      return Error::InvalidProgram;
    }
    const size_t n = static_cast<size_t>(t.numel());
    for (size_t i = 0; i < n; ++i) {
      if (!data[i]) return false;
    }
    return true;
  } else if (cond.isBool()) {
    return cond.toBool();
  }
  ET_LOG(
      Error,
      "parse_cond_value: predicate EValue is neither Bool nor Tensor[Bool]");
  return Error::InvalidProgram;
}

constexpr size_t kMaxHops = 10'000'000;

Error execute_step(DelegateInstance* d, const Step& step) {
  return std::visit(
      [&](auto&& s) -> Error {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, JumpFalseStep>) {
          ET_LOG(
              Error,
              "execute_step called with JumpFalseStep — this is a routing "
              "bug; jumps must be handled in the PC walker");
          return Error::Internal;
        } else if constexpr (std::is_same_v<T, MoveStep>) {
          if (s.src_value_id == s.dst_value_id) return Error::Ok;
          d->values[s.dst_value_id] = d->values[s.src_value_id];
          ET_LOG(
              Debug,
              "[cf] MoveStep src=%u -> dst=%u (EValue assign)",
              s.src_value_id,
              s.dst_value_id);
          return Error::Ok;
        } else {
          std::vector<Event*> waits_storage;
          waits_storage.reserve(s.wait_for.size());
          for (EventId id : s.wait_for) {
            if (Event* e = resolve_event(d, id)) waits_storage.push_back(e);
          }
          Span<Event* const> waits(waits_storage.data(), waits_storage.size());
          Event* signal = resolve_event(d, s.signal);

          if constexpr (std::is_same_v<T, ComputeStep>) {
            Engine* inst = d->plan.instances[s.runtime_idx];
            return inst->execute(
                s.segment,
                Span<EValue>(d->values.data(), d->values.size()),
                waits,
                signal);
          } else if constexpr (std::is_same_v<T, TransferStep>) {
            if (s.src_value_id >= d->values.size() ||
                s.dst_value_id >= d->values.size()) {
              return Error::InvalidState;
            }
            EValue& src_ev = d->values[s.src_value_id];
            EValue& dst_ev = d->values[s.dst_value_id];
            Engine* src_inst = d->plan.instances[s.src_idx];
            Engine* dst_inst = d->plan.instances[s.dst_idx];
            std::string src_pname = "?";
            std::string dst_pname = "?";
            if (s.src_idx < d->plan.providers.size() &&
                d->plan.providers[s.src_idx]) {
              src_pname = std::string(d->plan.providers[s.src_idx]->name());
            }
            if (s.dst_idx < d->plan.providers.size() &&
                d->plan.providers[s.dst_idx]) {
              dst_pname = std::string(d->plan.providers[s.dst_idx]->name());
            }
            size_t xfer_bytes =
                src_ev.isTensor() ? src_ev.toTensor().nbytes() : 0;
            ET_LOG(
                Debug,
                "[mem] step: TransferStep src=%u (%s) -> dst=%u (%s) bytes=%zu",
                s.src_value_id,
                src_pname.c_str(),
                s.dst_value_id,
                dst_pname.c_str(),
                xfer_bytes);
            // Direction-specific dispatch: the device (non-host) Engine
            // owns the cross-runtime move. Engine resolves its own
            // Buffer internally from the value_id.
            if (s.src_idx == kHostIdx && s.dst_idx != kHostIdx) {
              return dst_inst->upload_from_host(
                  src_ev, dst_ev, s.dst_value_id, waits, signal);
            } else if (s.dst_idx == kHostIdx && s.src_idx != kHostIdx) {
              return src_inst->download_to_host(
                  src_ev, s.src_value_id, dst_ev, waits, signal);
            } else {
              ET_LOG(
                  Error,
                  "TransferStep with neither side on host (src_idx=%u dst_idx=%u) is unsupported",
                  s.src_idx,
                  s.dst_idx);
              return Error::NotSupported;
            }
          }
        }
        return Error::Internal;
      },
      step);
}

} // namespace

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

    d->graph = std::make_unique<Graph>(plans->Get(0));

    // Borrow the process-wide registry (constructed lazily on first
    // call). DelegateInstance does not own Runtimes or RuntimeContexts;
    // it only owns the per-program Engines below.
    auto& registry = native_runtime_registry();

    auto avail = registry.available();
    d->owned_instances.reserve(avail.size());
    std::vector<Engine*> raw_instances;
    raw_instances.reserve(avail.size());
    std::vector<Runtime*> raw_providers;
    raw_providers.reserve(avail.size());
    for (auto* p : avail) {
      auto inst = p->instantiate();
      raw_instances.push_back(inst.get());
      raw_providers.push_back(p);
      d->owned_instances.push_back(std::move(inst));
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

    size_t num_orig = d->graph->num_values();
    size_t total_size = num_orig;
    for (const auto& mv : d->plan.mirror_values) {
      total_size = std::max<size_t>(total_size, mv.mirror_id + 1);
    }
    d->values.reserve(total_size);
    d->values.resize(total_size);
    if (auto e = initialize_values(d, runtime_alloc); e != Error::Ok) {
      d->~DelegateInstance();
      return e;
    }
    if (auto e = materialize_mirror_values(d); e != Error::Ok) {
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

    // Precompute mirror_index for fast IO bucketing.
    for (const auto& mv : d->plan.mirror_values) {
      RuntimeIndex eng = (mv.mirror_id < d->value_owner.size())
          ? d->value_owner[mv.mirror_id] : kInvalidRuntimeIdx;
      if (eng == kInvalidRuntimeIdx) continue;
      d->mirror_index[mv.source_value_id].push_back(
          {mv.mirror_id, eng});
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
      if (++hops > kMaxHops) {
        ET_LOG(
            Error,
            "NativeBackend: kMaxHops exceeded — malformed back edge?");
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
          if (id >= d->plan.events.size()) continue;
          auto& slot = d->plan.events[id];
          if (slot.event && slot.owner) {
            Error e = slot.owner->wait(slot.event.get());
            if (e != Error::Ok && first_err == Error::Ok) first_err = e;
          }
        }
        if (first_err != Error::Ok) break;

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
      if (auto* cs = std::get_if<ComputeStep>(&step)) sig = cs->signal;
      else if (auto* ts = std::get_if<TransferStep>(&step)) sig = ts->signal;
      if (sig != kNoEvent) observed_signals.push_back(sig);

      ++pc;
    }

    std::unordered_set<EventId> observed_set(
        observed_signals.begin(), observed_signals.end());
    for (EventId id : d->plan.terminal_events) {
      if (id >= d->plan.events.size()) continue;
      if (observed_set.count(id) == 0) continue;
      auto& slot = d->plan.events[id];
      if (slot.event && slot.owner) {
        Error e = slot.owner->wait(slot.event.get());
        if (e != Error::Ok && first_err == Error::Ok) first_err = e;
      }
    }

    if (first_err != Error::Ok) {
      d->poisoned = true;
      return first_err;
    }

    // Per-output post-execute pass (unchanged from before).
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
          std::memcpy(dp, sp, host_t.nbytes());
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

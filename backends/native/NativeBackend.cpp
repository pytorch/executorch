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

#include <executorch/backends/native/core/EngineUtils.h>
#include <executorch/backends/native/core/Router.h>
#include <executorch/backends/native/core/RuntimeRegistry.h>
#include <executorch/backends/native/ir/Plan.h>
#include <executorch/backends/native/ir/Step.h>
#include <executorch/backends/native/routers/GreedyRouter.h>
#include <executorch/backends/native/runtimes/cpu/CpuRuntime.h>
#include <executorch/backends/native/runtimes/host_pool/HostPoolRuntime.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#ifdef ET_NATIVE_HAS_METAL
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
#include <sstream>
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
  // value->Buffer table.
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

// Build the default ProviderSet. Only the host pool is registered
// eagerly — it's cheap, has no device init, and is needed by every
// program (it owns graph IO and cross-runtime intermediates).
//
// Compute providers (CPU, Metal, fake_accel) are constructed LAZILY by
// the per-init lazy_*_runtime() singletons below, and only when the
// load-time `compute_unit` option allows them. This avoids paying
// device init costs (MTLDevice creation, etc.) for loads that don't
// use that compute unit.
std::vector<std::unique_ptr<Runtime>> make_default_runtimes() {
  std::vector<std::unique_ptr<Runtime>> ps;
  ps.push_back(std::make_unique<HostPoolRuntime>());
  return ps;
}

// Lazy CPU singleton. Always available; the only conditioning is the
// load-time `compute_unit` option (caller decides whether to invoke).
Runtime* lazy_cpu_runtime() {
  static auto cpu = std::make_unique<CpuRuntime>();
  return cpu.get();
}

// Lazy "fake_accel" singleton — a CpuRuntime restricted to add/mul
// for routing tests. Opted in via compute_unit="fake_accel".
Runtime* lazy_fake_accel_runtime() {
  static auto fake = []() {
    std::unordered_set<std::string> allow = {
        "aten::add",
        "aten::add.Tensor",
        "aten::mul",
        "aten::mul.Tensor",
    };
    auto r = std::make_unique<CpuRuntime>("fake_accel", std::move(allow));
    ET_LOG(
        Info,
        "NativeBackend: registering CpuRuntime(\"fake_accel\") (lazy; routing-test scaffolding)");
    return r;
  }();
  return fake.get();
}

#ifdef ET_NATIVE_HAS_METAL
// Lazy Metal singleton. Constructed (and the MTLDevice initialized) on
// the first init() that allows Metal via compute_unit. Returns nullptr
// if Metal compiled in but its stream couldn't initialize (no Metal-
// capable device).
//
// Function-local static defers construction until the first call
// (Meyers singleton; thread-safe in C++11+). Once constructed, the
// MetalRuntime lives until process exit.
Runtime* lazy_metal_runtime() {
  static std::unique_ptr<MetalRuntime> metal =
      []() -> std::unique_ptr<MetalRuntime> {
    auto m = std::make_unique<MetalRuntime>();
    if (!m->stream_ready()) {
      ET_LOG(Info, "NativeBackend: MetalRuntime unavailable");
      return nullptr;
    }
    ET_LOG(Info, "NativeBackend: registering MetalRuntime (lazy)");
    return m;
  }();
  return metal.get();
}
#endif

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
            runtime_alloc
                ->allocateList<std::optional<::executorch::aten::Tensor>>(cnt);
        if (!evalp_list || !opt_list) {
          return Error::MemoryAllocationFailed;
        }
        EValue* none_ev = nullptr;
        for (size_t j = 0; j < cnt; ++j) {
          int32_t vidx = items[j];
          if (vidx == -1) {
            if (!none_ev) {
              void* mem = runtime_alloc->allocateInstance<EValue>();
              if (!mem)
                return Error::MemoryAllocationFailed;
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

// Drives each provider's allocate_buffers from the per-provider
// alloc_plan the router emitted. Engines internally allocate Buffers,
// populate their value->Buffer tables, and (for host-addressable
// Buffers) write data_ptr onto the central EValue array. NativeBackend
// keeps no per-vid bookkeeping.
//
// Validates per-claim that DeviceMirror / DeviceOnly requests were
// Claimed; declining one is a contract violation (router routed a
// dynamic-shape vid to a non-dynamic engine, or engine misbehaved).
// Init fails immediately with a descriptive error.
Error materialize_buffers(DelegateInstance* d) {
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

    // Validate: DeviceMirror / DeviceOnly requests are non-negotiable.
    // An engine that can't honor one (e.g., no resize_tensor support
    // for a mem_obj_id < 0 dynamic vid) must Decline so we can fail
    // init with a clear diagnostic rather than crash at execute time.
    for (size_t i = 0; i < reqs.size(); ++i) {
      const auto& req = reqs[i];
      bool requires_claim =
          (req.kind == MemoryKind::DeviceMirror ||
           req.kind == MemoryKind::DeviceOnly);
      if (requires_claim && claims[i] != Engine::AllocClaim::Claimed) {
        ET_LOG(
            Error,
            "materialize_buffers: provider %zu declined %s request for "
            "value_id=%u (mem_obj_id=%d). DeviceMirror/DeviceOnly are "
            "non-negotiable; the engine must Claim or fail allocate_buffers. "
            "If this is a mem_obj_id<0 dynamic vid, the engine likely lacks "
            "resize_tensor support and the router shouldn't have routed it here.",
            p,
            to_string(req.kind),
            req.value_id,
            req.mem_obj_id);
        return Error::NotSupported;
      }
    }
  }

  ET_LOG(Debug, "[mem] materialize_buffers: complete");
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
Error upload_constants(
    DelegateInstance* d,
    const ::executorch::runtime::NamedDataMap* ndm) {
  if (d->plan.const_plans.empty())
    return Error::Ok;
  // ndm is required only if any request needs NDM lookup. Inline-only
  // requests (ndm_key empty, inline_data set) don't touch the NDM.
  bool any_ndm_needed = false;
  for (const auto& reqs : d->plan.const_plans) {
    for (const auto& req : reqs) {
      if (!req.ndm_key.empty()) {
        any_ndm_needed = true;
        break;
      }
    }
    if (any_ndm_needed)
      break;
  }
  if (any_ndm_needed && !ndm) {
    ET_LOG(
        Error,
        "upload_constants: const_plans contain NDM-keyed requests but NamedDataMap is null");
    return Error::InvalidArgument;
  }
  for (size_t p = 0; p < d->plan.const_plans.size(); ++p) {
    const auto& reqs = d->plan.const_plans[p];
    if (reqs.empty())
      continue;
    if (p >= d->plan.instances.size() || !d->plan.instances[p]) {
      return Error::InvalidState;
    }
    Engine* inst = d->plan.instances[p];
    auto err = inst->upload_constants(
        ndm, Span<const Engine::ConstRequest>(reqs.data(), reqs.size()));
    if (err != Error::Ok) {
      ET_LOG(
          Error,
          "upload_constants: provider %zu (%s) failed (%zu requests)",
          p,
          d->plan.providers[p]
              ? std::string(d->plan.providers[p]->name()).c_str()
              : "?",
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
          d->plan.providers[p]
              ? std::string(d->plan.providers[p]->name()).c_str()
              : "?");
    }
  }
  return Error::Ok;
}

// Per-execute IO binding: fan out to every engine. Each engine
// self-filters via its internal io_*_bindings_ table built at
// init-time by set_io_bindings.
Error bind_inputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t in_count = std::min(n_in, args.size());
  Span<EValue* const> input_args(args.data(), in_count);
  for (Engine* inst : d->plan.instances) {
    if (!inst)
      continue;
    auto err = inst->bind_inputs(
        Span<EValue>(d->values.data(), d->values.size()), input_args);
    if (err != Error::Ok)
      return err;
  }
  return Error::Ok;
}

Error bind_outputs(DelegateInstance* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t n_out = d->plan.outputs.size();
  if (args.size() < n_in)
    return Error::InvalidArgument;
  size_t out_count = std::min(n_out, args.size() - n_in);
  Span<EValue* const> output_args(args.data() + n_in, out_count);
  for (Engine* inst : d->plan.instances) {
    if (!inst)
      continue;
    auto err = inst->bind_outputs(
        Span<EValue>(d->values.data(), d->values.size()), output_args);
    if (err != Error::Ok)
      return err;
  }
  return Error::Ok;
}

inline Event* resolve_event(DelegateInstance* d, EventId id) {
  if (id == kNoEvent || id >= d->plan.events.size())
    return nullptr;
  return d->plan.events[id].event.get();
}

constexpr RuntimeIndex kHostIdx = 0;

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
          if (s.src_value_id == s.dst_value_id)
            return Error::Ok;
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
            if (Event* e = resolve_event(d, id))
              waits_storage.push_back(e);
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
            // Buffer internally from the value_id. By host-canonical
            // invariant, the non-host side is always a DeviceEngine.
            if (s.src_idx == kHostIdx && s.dst_idx != kHostIdx) {
              return static_cast<DeviceEngine*>(dst_inst)->upload_from_host(
                  src_ev, dst_ev, s.dst_value_id, waits, signal);
            } else if (s.dst_idx == kHostIdx && s.src_idx != kHostIdx) {
              return static_cast<DeviceEngine*>(src_inst)->download_to_host(
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

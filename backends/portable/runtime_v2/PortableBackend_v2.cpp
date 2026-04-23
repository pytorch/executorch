/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * PortableBackend_v2 — ExecuTorch BackendInterface adapter for the v8.2
 * portable runtime architecture.
 *
 * Registers "PortableBackend_v2" with the runtime, separate from the
 * existing "PortableBackend" so the two can coexist during migration.
 *
 * See PORTABLE_BACKEND_API_PROPOSAL.md (§3, §6) for the architecture.
 */

#include <executorch/backends/portable/runtime_v2/api/BindingTable.h>
#include <executorch/backends/portable/runtime_v2/api/Plan.h>
#include <executorch/backends/portable/runtime_v2/api/ProviderRegistry.h>
#include <executorch/backends/portable/runtime_v2/api/Router.h>
#include <executorch/backends/portable/runtime_v2/api/Step.h>
#include <executorch/backends/portable/runtime_v2/cpu/CpuProvider.h>
#include <executorch/backends/portable/runtime_v2/routers/GreedyRouter.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

#ifdef PORTABLE_V2_HAS_METAL
#include <executorch/backends/portable/runtime_v2/metal/MetalProvider.h>
#endif

#include <executorch/backends/portable/runtime_v2/api/GraphTypes.h>

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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace executorch {
namespace backends {
namespace portable_v2 {

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
 * delegate. See LoadedDelegate in §6 of the design doc.
 */
struct LoadedDelegate {
  // Lifetime root of providers/instances/registry.
  std::unique_ptr<ProviderRegistry> registry;
  std::vector<std::unique_ptr<Instance>> owned_instances;

  // Parsed program — wrapped behind Graph. Downstream code reaches the
  // serialized program ONLY through `graph`; nothing else holds a raw
  // flatbuffer pointer. See §3 of PORTABLE_BACKEND_API_PROPOSAL.md.
  std::unique_ptr<Graph> graph;

  // Universal value array, indexed by value_id. Holds EValues for
  // everything (scalars, lists, tensors). For tensor EValues, the
  // TensorImpl::data_ptr is a denormalized cache of the bound Buffer's
  // host_ptr (kept consistent at bind time).
  std::vector<EValue> values;

  // Side table: tensor storage backings only.
  BindingTable bindings;

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

  ~LoadedDelegate() {
    // Drain (no-op for CPU; matters when GPU instances are present).
    for (auto* inst : plan.instances) {
      if (inst) inst->drain();
    }
    // Release owned buffers via their owning Instance.
    for (auto& ob : plan.owned_buffers) {
      if (ob.owner && ob.buf) ob.owner->release_buffer(ob.buf);
    }
  }
};

// Build the default ProviderSet. v1: CpuProvider at index 0 (host slot).
// v1 supports at most ONE non-CPU provider per process. Selection rules
// (in order):
//   - If env PORTABLE_V2_USE_FAKE_ACCEL=1 → register FakeAccel CpuProvider
//     as second slot (test mode for routing). Skip Metal even on Apple.
//   - Else on Apple, when PORTABLE_V2_HAS_METAL is defined and
//     PORTABLE_V2_DISABLE_METAL is unset → try MetalProvider; register
//     iff the underlying MetalStream constructed successfully.
//   - Else → CpuProvider only (single-provider mode).
std::vector<std::unique_ptr<Provider>> make_default_providers() {
  std::vector<std::unique_ptr<Provider>> ps;
  ps.push_back(std::make_unique<CpuProvider>());

  const char* fake = std::getenv("PORTABLE_V2_USE_FAKE_ACCEL");
  if (fake && fake[0] == '1') {
    ET_LOG(Info,
           "PortableBackend_v2: registering second CpuProvider (\"fake_accel\") for routing tests");
    std::unordered_set<std::string> allow = {
        "aten::add", "aten::add.Tensor", "aten::mul", "aten::mul.Tensor",
    };
    ps.push_back(std::make_unique<CpuProvider>("fake_accel", std::move(allow)));
    return ps;
  }

#ifdef PORTABLE_V2_HAS_METAL
  const char* disable_metal = std::getenv("PORTABLE_V2_DISABLE_METAL");
  if (!disable_metal || disable_metal[0] != '1') {
    auto metal = std::make_unique<MetalProvider>();
    if (metal->stream_ready()) {
      ET_LOG(Info, "PortableBackend_v2: registering MetalProvider");
      ps.push_back(std::move(metal));
    } else {
      ET_LOG(Info,
             "PortableBackend_v2: MetalProvider unavailable; CPU-only mode");
    }
  }
#endif
  return ps;
}

// Initialize one tensor EValue from the flatbuffer Tensor metadata.
// Build a TensorImpl + EValue for value_id from the Graph adapter's
// typed metadata. The data_ptr starts at nullptr; the router/executor
// sets it later from the bound Buffer.
Error initialize_tensor_evalue(LoadedDelegate* d, uint32_t value_id) {
  const auto& graph = *d->graph;
  auto sizes = graph.tensor_sizes(value_id);
  auto dim_order_in = graph.tensor_dim_order(value_id);
  size_t dim = sizes.size();

  LoadedDelegate::TensorMeta meta;
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
  if (status != Error::Ok) return status;

  ScalarType dtype = graph.tensor_dtype(value_id);
  meta.impl.reset(new TensorImpl(
      dtype,
      static_cast<ssize_t>(dim),
      meta.sizes.get(),
      /*data=*/nullptr,
      meta.dim_order.get(),
      meta.strides.get(),
      // DYNAMIC_BOUND so kernels' resize_tensor() calls take effect.
      // The flatbuffer's `sizes` is the AOT memory-plan max bound;
      // runtime kernels may shrink to actual.
      ::executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND));

  d->values[value_id] = EValue(Tensor(meta.impl.get()));
  d->tensor_metas.push_back(std::move(meta));
  return Error::Ok;
}

// Initialize all value EValues. After this returns, every value_id in
// the graph has a populated EValue:
//   - Tensors: TensorImpl with dtype/sizes/dim_order/strides; data_ptr
//     starts as nullptr and is set later by prebind_owned_buffers /
//     bind_inputs / bind_outputs.
//   - Scalars / lists: actual values from the flatbuffer.
//   - Inputs/outputs are also materialized here as shell tensor EValues
//     using the flatbuffer-declared (max-shape) sizes, so allocate_all
//     can read their tensor metadata.
Error initialize_values(
    LoadedDelegate* d,
    ::executorch::runtime::MemoryAllocator* runtime_alloc) {
  using ::executorch::runtime::BoxedEvalueList;

  size_t n = d->graph->num_values();
  if (n == 0) return Error::Ok;

  // d->values is already sized by the init() caller. Don't shrink.
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
        // Materialize EVERY tensor (including IO). For IO, this creates
        // a shell TensorImpl with the flatbuffer-declared (max-shape)
        // sizes; bind_inputs/bind_outputs will resize per execute and
        // upload_from_host will re-alias the bound Buffer's data_ptr.
        if (auto e = initialize_tensor_evalue(d, i); e != Error::Ok) {
          return e;
        }
        break;
      case ValueType::IntList: {
        // Mirror Method::init's BoxedEvalueList<int64_t> setup
        // (see runtime/executor/method.cpp).
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
        if (!boxed_mem) return Error::MemoryAllocationFailed;
        auto* boxed = new (boxed_mem)
            BoxedEvalueList<int64_t>(evalp_list, int_list, cnt);
        d->values[i] = EValue(boxed);
        ET_LOG(Debug,
               "initialize_values: value_id=%u IntList(%zu items)", i, cnt);
      } break;
      case ValueType::Other:
        // String / OptionalTensor / BoolList / DoubleList / etc.
        // Adapter doesn't surface these yet; default-construct.
        d->values[i] = EValue();
        break;
    }
  }
  return Error::Ok;
}

// Pre-bind tensor EValues to their owned Buffers. The router populates
// `Plan::owned_buffers` with per-Instance allocations, constants, AND
// cross-runtime transfer destinations (which use synthetic value_ids).
// We walk those and bind each (value_id -> Buffer*) into the BindingTable
// AND sync the EValue's TensorImpl::data_ptr to the Buffer's host_ptr.
//
// Pre-condition: synthetic value EValues have already been materialized by
// materialize_synthetic_values (called before this).
void prebind_owned_buffers(LoadedDelegate* d) {
  for (auto& ob : d->plan.owned_buffers) {
    if (!ob.buf) continue;
    d->bindings.bind(ob.value_id, ob.buf);

    // Look up the owning Provider's name for clearer logs.
    std::string prov_name = "?";
    for (size_t i = 0; i < d->plan.instances.size(); ++i) {
      if (d->plan.instances[i] == ob.owner) {
        prov_name = std::string(d->plan.providers[i]->name());
        break;
      }
    }
    if (ob.value_id < d->values.size() &&
        d->values[ob.value_id].isTensor()) {
      void* hp = ob.buf->host_ptr();
      if (hp) {
        d->values[ob.value_id]
            .toTensor()
            .unsafeGetTensorImpl()
            ->set_data(hp);
      }
      ET_LOG(Debug,
             "[mem] bind: value_id=%u provider=%s host_ptr=%p bytes=%zu (tensor)",
             ob.value_id, prov_name.c_str(), hp, ob.buf->size_bytes());
    } else {
      ET_LOG(Debug,
             "[mem] bind: value_id=%u provider=%s bytes=%zu (non-tensor or unset)",
             ob.value_id, prov_name.c_str(), ob.buf->size_bytes());
    }
  }
}

// Materialize EValues for router-synthesized value_ids (cross-runtime
// transfer destinations). Each synthesized id inherits dtype/shape/strides
// from its source value_id by cloning TensorImpl metadata. The data_ptr
// is set later by prebind_owned_buffers when we know which Buffer.
//
// Pre-condition: d->values has already been sized to fit max(synthetic id)+1
// (done in init() before initialize_values to avoid invalidating
// BoxedEvalueList<int64_t> EValue* pointers).
Error materialize_synthetic_values(LoadedDelegate* d) {
  if (d->plan.synthetic_values.empty()) return Error::Ok;

  for (const auto& sv : d->plan.synthetic_values) {
    if (sv.new_id >= d->values.size()) {
      ET_LOG(Error,
             "materialize_synthetic_values: new_id=%u >= values.size()=%zu",
             sv.new_id, d->values.size());
      return Error::InvalidState;
    }
    if (sv.source_id >= d->values.size() ||
        !d->values[sv.source_id].isTensor()) {
      ET_LOG(Error,
             "materialize_synthetic_values: source value_id=%u is not a tensor",
             sv.source_id);
      return Error::InvalidProgram;
    }

    auto& src = d->values[sv.source_id].toTensor();
    auto* src_impl = src.unsafeGetTensorImpl();
    size_t dim = src.dim();

    LoadedDelegate::TensorMeta tm;
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
        // DYNAMIC_BOUND so per-execute TransferStep's resize_tensor()
        // can update sizes when actual shape varies per execute.
        ::executorch::aten::TensorShapeDynamism::DYNAMIC_BOUND));

    d->values[sv.new_id] = EValue(Tensor(tm.impl.get()));
    d->tensor_metas.push_back(std::move(tm));
    ET_LOG(Debug,
           "materialize_synthetic_values: new_id=%u (clone of source_id=%u)",
           sv.new_id, sv.source_id);
  }
  return Error::Ok;
}

// Single host-first allocation pass. Walks each provider's alloc_plan in
// provider index order (host slot 0 first). For each provider, patches
// host_alias on synthetic AllocRequests to point at the source value's
// (already-allocated) Buffer, then calls allocate_all. Pushes resulting
// Buffers into plan.owned_buffers and updates the value→Buffer ledger
// for the next provider.
//
// Pre-condition: initialize_values + materialize_synthetic_values have
// run (every value_id has an EValue with valid TensorImpl metadata).
Error allocate_buffers(LoadedDelegate* d) {
  // Build a value_id → source value_id map for synthetic mirrors.
  std::unordered_map<uint32_t, uint32_t> synth_to_source;
  for (const auto& sv : d->plan.synthetic_values) {
    synth_to_source[sv.new_id] = sv.source_id;
  }

  // value_id → Buffer ledger; populated as we allocate.
  std::unordered_map<uint32_t, Buffer*> value_to_buf;
  // Seed with constants already in plan.owned_buffers (uploaded during
  // route() via upload_constant).
  for (const auto& ob : d->plan.owned_buffers) {
    value_to_buf[ob.value_id] = ob.buf;
  }

  // Iterate providers in plan order (host slot 0 first). Earlier providers'
  // Buffers become visible as host_alias targets for later providers'
  // synthetic AllocRequests.
  for (size_t p = 0; p < d->plan.alloc_plans.size(); ++p) {
    auto& reqs = d->plan.alloc_plans[p];
    if (reqs.empty()) continue;
    if (p >= d->plan.instances.size() || !d->plan.instances[p]) {
      return Error::InvalidState;
    }
    Instance* inst = d->plan.instances[p];

    // Patch host_alias on synthetic AllocRequests.
    for (auto& req : reqs) {
      auto sit = synth_to_source.find(req.value_id);
      if (sit == synth_to_source.end()) continue;
      auto vit = value_to_buf.find(sit->second);
      if (vit == value_to_buf.end() || !vit->second) {
        ET_LOG(Error,
               "allocate_buffers: synthetic value_id=%u source value_id=%u "
               "not yet allocated (provider %zu allocates before its sources)",
               req.value_id, sit->second, p);
        return Error::InvalidState;
      }
      req.host_alias = vit->second;
    }

    // Call allocate_all.
    std::vector<Buffer*> out(reqs.size(), nullptr);
    auto e = inst->allocate_all(
        Span<const Instance::AllocRequest>(reqs.data(), reqs.size()),
        Span<const EValue>(d->values.data(), d->values.size()),
        Span<Buffer*>(out.data(), out.size()));
    if (e != Error::Ok) return e;

    // Record results.
    for (size_t i = 0; i < reqs.size(); ++i) {
      d->plan.owned_buffers.push_back({out[i], inst, reqs[i].value_id});
      value_to_buf[reqs[i].value_id] = out[i];
    }
  }
  return Error::Ok;
}

// bind_inputs: per-execute. For each graph input value_id:
//   - Overwrites d->values[vid] with caller's EValue (shares caller's
//     TensorImpl, so per-execute kernel resize on our slot is visible
//     to caller).
//   - Calls upload_from_host on the pre-allocated destination Buffer
//     to re-alias it to caller's pointer.
//
// Note: the shell TensorImpl materialized at init time (in initialize_values)
// served its purpose at allocate_all (provided dtype/sizes). Per-execute
// it's superseded by caller's TensorImpl via the EValue copy. The shell
// stays alive in d->tensor_metas until ~LoadedDelegate.
Error bind_inputs(LoadedDelegate* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  for (size_t i = 0; i < n_in && i < args.size(); ++i) {
    const auto& ib = d->plan.inputs[i];
    uint32_t vid = ib.value_id;
    if (vid >= d->values.size()) return Error::InvalidArgument;

    // Replace shell EValue with caller's (shares TensorImpl).
    d->values[vid] = *args[i];

    if (!d->values[vid].isTensor()) continue;

    auto& tensor = d->values[vid].toTensor();
    void* host_data = tensor.mutable_data_ptr();
    size_t nbytes = tensor.nbytes();
    if (!host_data || nbytes == 0) continue;

    Buffer* buf = d->bindings.get(vid);
    if (!buf) {
      ET_LOG(Error,
             "bind_inputs: no pre-allocated Buffer for input value_id=%u",
             vid);
      return Error::InvalidState;
    }
    Instance* dst_inst = d->plan.instances[0];

    if (auto e = dst_inst->upload_from_host(
            d->values[vid], host_data,
            d->values[vid], buf,
            QueueKind::Compute,
            Span<Event* const>(),
            nullptr);
        e != Error::Ok) {
      return e;
    }
    ET_LOG(Debug,
           "[mem] bind_input: value_id=%u alias_into=cpu(host) caller_ptr=%p bytes=%zu",
           vid, host_data, nbytes);
  }
  return Error::Ok;
}

// bind_outputs: symmetric.
Error bind_outputs(LoadedDelegate* d, Span<EValue*> args) {
  size_t n_in = d->plan.inputs.size();
  size_t n_out = d->plan.outputs.size();
  for (size_t i = 0; i < n_out && (n_in + i) < args.size(); ++i) {
    const auto& ob = d->plan.outputs[i];
    uint32_t vid = ob.value_id;
    if (vid >= d->values.size()) return Error::InvalidArgument;

    EValue* arg_ev = args[n_in + i];
    if (!arg_ev) continue;

    d->values[vid] = *arg_ev;
    // Producer-side graph-output mirrors (synthetic value_ids whose
    // source is this output) must share the caller's TensorImpl so
    // that:
    //   (a) the producer kernel writes to caller's data_ptr, and
    //   (b) resize_tensor calls inside the kernel update caller's
    //       TensorImpl (the one our caller will read sizes from).
    // The TransferStep host -> producer (emitted by the router for
    // each producer-side mirror) handles re-aliasing the producer's
    // Buffer to caller's pointer per-execute.
    for (const auto& sv : d->plan.synthetic_values) {
      if (sv.source_id == vid && sv.new_id < d->values.size()) {
        d->values[sv.new_id] = *arg_ev;
      }
    }
    if (!d->values[vid].isTensor()) continue;

    auto& tensor = d->values[vid].toTensor();
    void* host_data = tensor.mutable_data_ptr();
    size_t nbytes = tensor.nbytes();
    if (!host_data || nbytes == 0) continue;

    Buffer* buf = d->bindings.get(vid);
    if (!buf) {
      ET_LOG(Error,
             "bind_outputs: no pre-allocated Buffer for output value_id=%u",
             vid);
      return Error::InvalidState;
    }
    Instance* dst_inst = d->plan.instances[0];

    if (auto e = dst_inst->upload_from_host(
            d->values[vid], host_data,
            d->values[vid], buf,
            QueueKind::Compute,
            Span<Event* const>(),
            nullptr);
        e != Error::Ok) {
      return e;
    }
    ET_LOG(Debug,
           "[mem] bind_output: value_id=%u alias_into=cpu(host) caller_ptr=%p bytes=%zu",
           vid, host_data, nbytes);
  }
  return Error::Ok;
}

// Execute one Step.
Error execute_step(LoadedDelegate* d, const Step& step) {
  // Pre-resolve wait_for / signal Event*. v1 has no events emitted by the
  // router, so wait_for is empty and signal is kNoEvent. Keep the
  // resolution code shape so adding events later is mechanical.
  std::vector<Event*> waits_storage;
  Span<Event* const> waits(waits_storage.data(), static_cast<size_t>(0));
  Event* signal = nullptr;

  return std::visit(
      [&](auto&& s) -> Error {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, ComputeStep>) {
          // Resolve wait_for from EventIds (skipped in v1; empty).
          (void)s.wait_for;
          (void)s.signal;
          Instance* inst = d->plan.instances[s.runtime_idx];
          return inst->execute(
              s.segment,
              Span<EValue>(d->values.data(), d->values.size()),
              d->bindings,
              waits,
              signal);
        } else if constexpr (std::is_same_v<T, TransferStep>) {
          if (s.src_value_id >= d->values.size() ||
              s.dst_value_id >= d->values.size()) {
            return Error::InvalidState;
          }
          EValue& src_ev = d->values[s.src_value_id];
          EValue& dst_ev = d->values[s.dst_value_id];
          Buffer* src_buf = d->bindings.get(s.src_value_id);
          Buffer* dst_buf = d->bindings.get(s.dst_value_id);
          Instance* src_inst = d->plan.instances[s.src_idx];
          Instance* dst_inst = d->plan.instances[s.dst_idx];
          // Per-execute trace line: ties the runtime transfer back to
          // the router's "[mem] router: cross-runtime transfer ..."
          // setup line so memory cost is auditable.
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
          size_t xfer_bytes = src_ev.isTensor() ? src_ev.toTensor().nbytes() : 0;
          ET_LOG(Debug,
                 "[mem] step: TransferStep src_value_id=%u (%s) -> dst_value_id=%u (%s) bytes=%zu",
                 s.src_value_id, src_pname.c_str(),
                 s.dst_value_id, dst_pname.c_str(),
                 xfer_bytes);
          // Direction-specific dispatch: the device (non-host) Instance
          // owns the cross-runtime move. By convention slot 0 is host
          // (CPU), so whichever side != 0 is the device side.
          //   src is host → upload to device's Buffer
          //   dst is host → download from device's Buffer
          // Two non-host runtimes is not supported (and never emitted by
          // the router); a runtime↔runtime transfer would route through
          // host as two TransferSteps.
          constexpr RuntimeIndex kHostIdx = 0;
          if (s.src_idx == kHostIdx && s.dst_idx != kHostIdx) {
            void* host_src_ptr = src_ev.isTensor()
                ? src_ev.toTensor().mutable_data_ptr()
                : nullptr;
            return dst_inst->upload_from_host(
                src_ev, host_src_ptr,
                dst_ev, dst_buf,
                s.queue, waits, signal);
          } else if (s.dst_idx == kHostIdx && s.src_idx != kHostIdx) {
            void* host_dst_ptr = dst_ev.isTensor()
                ? dst_ev.toTensor().mutable_data_ptr()
                : nullptr;
            return src_inst->download_to_host(
                src_ev, src_buf,
                dst_ev, host_dst_ptr,
                s.queue, waits, signal);
          } else {
            ET_LOG(Error,
                   "TransferStep with neither side on host (src_idx=%u dst_idx=%u) is unsupported",
                   s.src_idx, s.dst_idx);
            return Error::NotSupported;
          }
        }
        return Error::Internal;
      },
      step);
}

}  // namespace

class PortableBackendV2 final : public ::executorch::runtime::BackendInterface {
 public:
  ~PortableBackendV2() override = default;

  bool is_available() const override { return true; }

  Result<DelegateHandle*> init(
      BackendInitContext& ctx,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> /*compile_specs*/) const override {
    if (!processed || !processed->data() || processed->size() == 0) {
      return Error::InvalidArgument;
    }

    auto* program = executorch_flatbuffer::GetProgram(processed->data());
    if (!program) return Error::InvalidProgram;

    auto plans = program->execution_plan();
    if (!plans || plans->size() == 0) return Error::InvalidProgram;

    // Allocate the LoadedDelegate from the runtime allocator (lifetime =
    // method). We placement-new because MemoryAllocator::allocate returns
    // raw bytes.
    auto* runtime_alloc = ctx.get_runtime_allocator();
    if (!runtime_alloc) return Error::InvalidState;
    void* mem = runtime_alloc->allocate(
        sizeof(LoadedDelegate), alignof(LoadedDelegate));
    if (!mem) return Error::MemoryAllocationFailed;
    auto* d = new (mem) LoadedDelegate();

    // Wrap the parsed flatbuffer in a Graph immediately. From this point
    // on, NO downstream code (Router, Instances, initialize_values,
    // materialize_synthetic_values, prebind_owned_buffers, executor) reaches
    // into the flatbuffer directly — Graph is the only handle.
    d->graph = std::make_unique<Graph>(plans->Get(0));

    // Build the provider set and registry.
    d->registry =
        std::make_unique<ProviderRegistry>(make_default_providers());

    // Instantiate one Instance per available provider.
    auto avail = d->registry->available();
    d->owned_instances.reserve(avail.size());
    std::vector<Instance*> raw_instances;
    raw_instances.reserve(avail.size());
    std::vector<Provider*> raw_providers;
    raw_providers.reserve(avail.size());
    for (auto* p : avail) {
      auto inst = p->instantiate();
      raw_instances.push_back(inst.get());
      raw_providers.push_back(p);
      d->owned_instances.push_back(std::move(inst));
    }

    // Route FIRST so we know the synthetic-value count before sizing
    // d->values. (initialize_values builds BoxedEvalueList<int64_t>'s
    // that store EValue* pointers into d->values; if we resize() after
    // that, those pointers are invalidated.)
    GreedyRouter router;
    RouterOptions opts;
    auto plan_result = router.route(
        *d->graph,
        Span<Provider* const>(raw_providers.data(), raw_providers.size()),
        Span<Instance* const>(raw_instances.data(), raw_instances.size()),
        ctx.get_named_data_map(),
        opts);
    if (!plan_result.ok()) {
      d->~LoadedDelegate();
      return plan_result.error();
    }
    d->plan = std::move(plan_result.get());

    // Pre-size d->values to fit both original and synthetic value_ids.
    // Reserve THEN resize to the final size up-front: subsequent
    // initialize_values builds BoxedEvalueList<int64_t>'s that store
    // EValue* pointers into d->values; if the vector reallocates later,
    // those pointers dangle.
    size_t num_orig = d->graph->num_values();
    size_t total_size = num_orig;
    for (const auto& sv : d->plan.synthetic_values) {
      total_size = std::max<size_t>(total_size, sv.new_id + 1);
    }
    d->values.reserve(total_size);
    d->values.resize(total_size);

    // Materialize EValues for all original value_ids (including IO with
    // shell TensorImpls at flatbuffer-declared sizes).
    if (auto e = initialize_values(d, runtime_alloc); e != Error::Ok) {
      d->~LoadedDelegate();
      return e;
    }
    // Materialize EValues for synthetic value_ids (clone of source's
    // TensorImpl; data_ptr null until allocate_buffers + prebind).
    if (auto e = materialize_synthetic_values(d); e != Error::Ok) {
      d->~LoadedDelegate();
      return e;
    }
    // Now allocate Buffers for everything via allocate_all (host-first
    // single pass; synthetic AllocRequests' host_alias gets patched to
    // the source's just-allocated Buffer before each device call).
    if (auto e = allocate_buffers(d); e != Error::Ok) {
      d->~LoadedDelegate();
      return e;
    }
    // Bind allocated Buffers into the BindingTable + sync TensorImpl
    // data_ptr to each Buffer's host_ptr.
    prebind_owned_buffers(d);

    ET_LOG(Info, "PortableBackend_v2: initialized with %zu steps",
           d->plan.steps.size());
    return reinterpret_cast<DelegateHandle*>(d);
  }

  Error execute(
      BackendExecutionContext& /*ctx*/,
      DelegateHandle* handle,
      Span<EValue*> args) const override {
    auto* d = static_cast<LoadedDelegate*>(handle);
    if (!d) return Error::InvalidState;

    // Failure model: a Failed/Poisoned event from any prior execute
    // leaves the delegate sticky-Poisoned. Subsequent calls bail.
    if (d->poisoned) return Error::Internal;  // DelegatePoisoned

    // Bindings are persistent (allocated once at init); per-execute
    // bind_inputs / bind_outputs re-aliases the existing Buffers in
    // place via upload_from_host.

    if (auto e = bind_inputs(d, args); e != Error::Ok) return e;
    if (auto e = bind_outputs(d, args); e != Error::Ok) return e;

    // Issue every step. Status flows through events; we ignore the
    // return value of each visit so poison can propagate. (v1 has no
    // events, so this is just sequencing.)
    Error first_err = Error::Ok;
    for (const auto& step : d->plan.steps) {
      Error e = execute_step(d, step);
      if (e != Error::Ok && first_err == Error::Ok) {
        first_err = e;
      }
    }

    // Single drain at end.
    for (auto* inst : d->plan.instances) {
      if (inst) inst->drain();
    }

    if (first_err != Error::Ok) {
      d->poisoned = true;
      return first_err;
    }

    // Copy back scalar outputs (rare). Tensor outputs were written
    // in-place into caller storage by either the producing kernel (host
    // outputs) or a terminal device->host TransferStep.
    size_t n_in = d->plan.inputs.size();
    size_t n_out = d->plan.outputs.size();
    for (size_t i = 0; i < n_out && (n_in + i) < args.size(); ++i) {
      const auto& ob = d->plan.outputs[i];
      uint32_t vid = ob.value_id;
      if (vid >= d->values.size()) continue;
      EValue* arg_ev = args[n_in + i];
      if (!arg_ev) continue;
      if (!d->values[vid].isTensor()) {
        *arg_ev = d->values[vid];
      }
    }
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle) const override {
    if (!handle) return;
    auto* d = static_cast<LoadedDelegate*>(handle);
    d->~LoadedDelegate();
    // The underlying memory was bump-allocated from the runtime
    // allocator and is reclaimed when the Method is destroyed; no free
    // here.
  }
};

namespace {
PortableBackendV2 g_backend_v2;
Backend g_backend_record{"PortableBackend_v2", &g_backend_v2};
auto g_register =
    ::executorch::runtime::register_backend(g_backend_record);
}  // namespace

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

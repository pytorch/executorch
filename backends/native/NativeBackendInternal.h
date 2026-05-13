/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * Internal shared header for the NativeBackend translation units.
 *
 * NativeBackend.cpp was split into several sibling .cpp files
 * (NativeBackendRegistry.cpp, NativeBackendValueInit.cpp,
 * NativeBackendBuffers.cpp, NativeBackendExecute.cpp). This header
 * carries the cross-file types, using-declarations, and free-function
 * prototypes those files share. Not part of any public API — strictly
 * implementation-internal.
 */

#include <executorch/backends/native/core/EngineUtils.h>
#include <executorch/backends/native/core/Router.h>
#include <executorch/backends/native/core/RuntimeRegistry.h>
#include <executorch/backends/native/ir/GraphTypes.h>
#include <executorch/backends/native/ir/Plan.h>
#include <executorch/backends/native/ir/Step.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

#include <memory>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace native {

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
using ::executorch::runtime::Span;
using ValueType = ::executorch::backends::portable::ValueType;

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
 * Note: the RuntimeRegistry (and the Runtimes it owns) lives at process
 * scope, not on this struct — see native_runtime_registry() in
 * NativeBackendRegistry.cpp. Only the per-program Engines
 * (`owned_instances`) and routing state (`graph`, `plan`, `values`,
 * etc.) are owned here.
 */
struct DelegateInstance {
  // Per-program Engines. One per available process-wide Runtime,
  // produced by Runtime::instantiate(graph) at init. Each Engine
  // holds its Runtime's typed shared state (e.g., Metal stream)
  // directly via constructor injection from its concrete Runtime.
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

// Host runtime always lives at slot 0 in candidate_providers. The
// canonical definition lives in core/RuntimeId.h (included transitively
// via Plan.h / Step.h); referenced here so callers don't need to know
// the deeper header to use it.

// ---- Cross-file free-function prototypes -------------------------------

// NativeBackendRegistry.cpp
::executorch::backends::native::RuntimeRegistry& native_runtime_registry();
Runtime* lazy_cpu_runtime();
Runtime* lazy_fake_accel_runtime();
#ifdef ET_NATIVE_HAS_METAL
Runtime* lazy_metal_runtime();
#endif

// NativeBackendValueInit.cpp
Error initialize_tensor_evalue(DelegateInstance* d, uint32_t value_id);
Error initialize_values(
    DelegateInstance* d,
    ::executorch::runtime::MemoryAllocator* runtime_alloc);

// NativeBackendBuffers.cpp
Error materialize_buffers(DelegateInstance* d);
Error upload_constants(
    DelegateInstance* d,
    const ::executorch::runtime::NamedDataMap* ndm);

// NativeBackendExecute.cpp
Error bind_inputs(DelegateInstance* d, Span<EValue*> args);
Error bind_outputs(DelegateInstance* d, Span<EValue*> args);
Event* resolve_event(DelegateInstance* d, EventId id);
Error execute_step(DelegateInstance* d, const Step& step);

} // namespace native
} // namespace backends
} // namespace executorch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/weight_offload/session.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>

#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/cuda_delegate_handle.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/weight_offload/probe_registry.h>

// Out-of-line ``CudaDelegateHandle`` destructor lives here so the
// implicit ``unique_ptr<Session>`` member destructor sees Session
// complete.
namespace executorch::backends::cuda {
CudaDelegateHandle::~CudaDelegateHandle() = default;
} // namespace executorch::backends::cuda

namespace executorch::backends::cuda::weight_offload {

// ---------------------------------------------------------------------------
// DummyInstallation — RAII for the 1-byte dummies + SlimTensor wrappers
// AOTI dispatches probes with. Defined out-of-line because the destructor
// touches CUDA + the SlimTensor type, and inlining either drags includes
// into every TU that pulls session.h.
// ---------------------------------------------------------------------------

DummyInstallation::~DummyInstallation() {
  clear_and_free();
}

DummyInstallation::DummyInstallation(DummyInstallation&& other) noexcept
    : dummy_data_ptrs(std::move(other.dummy_data_ptrs)),
      dummy_tensors(std::move(other.dummy_tensors)) {
  other.dummy_data_ptrs.clear();
  other.dummy_tensors.clear();
}

DummyInstallation& DummyInstallation::operator=(
    DummyInstallation&& other) noexcept {
  if (this != &other) {
    clear_and_free();
    dummy_data_ptrs = std::move(other.dummy_data_ptrs);
    dummy_tensors = std::move(other.dummy_tensors);
    other.dummy_data_ptrs.clear();
    other.dummy_tensors.clear();
  }
  return *this;
}

void DummyInstallation::clear_and_free() {
  for (auto* t : dummy_tensors) {
    delete t;
  }
  dummy_tensors.clear();
  for (void* p : dummy_data_ptrs) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  dummy_data_ptrs.clear();
}

namespace {

// kCudaDeviceType comes from payload.h (in the
// executorch::backends::cuda::weight_offload namespace).

void free_host_mirror(
    std::unordered_map<std::string, Session::HostEntry>& host_entries) {
  for (auto& [fqn, entry] : host_entries) {
    if (entry.host_ptr != nullptr) {
      cudaFreeHost(entry.host_ptr);
      entry.host_ptr = nullptr;
    }
  }
  host_entries.clear();
}

// Resolved budget + the BudgetSpec describing the user's source
// (for the below-floor UX message).
struct ResolvedBudget {
  uint64_t budget_bytes{0};
  BudgetSpec spec{};
};

// Resolve the load-time budget from the runtime-spec chain.
// Tries the public ``weight_offload_budget_mb`` (int MB) first,
// then the internal ``_weight_offload_internal_budget_bytes``
// (decimal string of bytes), then defaults to ``floor + pinned``.
// Default-path BudgetSpec.name still points at the public spec so
// the UX hint reaches for the public knob.
::executorch::runtime::Result<ResolvedBudget> resolve_budget(
    ::executorch::runtime::BackendInitContext& context,
    uint64_t floor_bytes,
    uint64_t pinned_bytes_total) {
  using ::executorch::runtime::Error;
  ResolvedBudget out;
  out.spec.name = "weight_offload_budget_mb";
  out.spec.value_is_mb = true;

  // Public path: weight_offload_budget_mb (int megabytes).
  auto mb_res = context.get_runtime_spec<int>("weight_offload_budget_mb");
  if (mb_res.ok()) {
    int mb = mb_res.get();
    if (mb <= 0) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] weight_offload_budget_mb=%d "
          "must be a positive integer\n",
          mb);
      return Error::InvalidArgument;
    }
    out.budget_bytes = static_cast<uint64_t>(mb) << 20;
    out.spec.value = static_cast<uint64_t>(mb);
    return out;
  }
  if (mb_res.error() == Error::InvalidArgument) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] weight_offload_budget_mb is set "
        "but not an int\n");
    return Error::InvalidArgument;
  }
  if (mb_res.error() != Error::NotFound) {
    return mb_res.error();
  }

  // Internal byte spec — for exact-byte test paths.
  auto bytes_res = context.get_runtime_spec<const char*>(
      "_weight_offload_internal_budget_bytes");
  if (bytes_res.ok()) {
    const char* spec_cstr = bytes_res.get();
    if (spec_cstr == nullptr || spec_cstr[0] == '\0') {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] "
          "_weight_offload_internal_budget_bytes is empty\n");
      return Error::InvalidArgument;
    }
    if (spec_cstr[0] == '-' || spec_cstr[0] == '+') {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] "
          "_weight_offload_internal_budget_bytes='%s' has a sign\n",
          spec_cstr);
      return Error::InvalidArgument;
    }
    char* end = nullptr;
    errno = 0;
    unsigned long long parsed = std::strtoull(spec_cstr, &end, 10);
    if (errno != 0 || end == spec_cstr || *end != '\0' || parsed == 0) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] "
          "_weight_offload_internal_budget_bytes='%s' must be a "
          "positive decimal integer\n",
          spec_cstr);
      return Error::InvalidArgument;
    }
    out.budget_bytes = static_cast<uint64_t>(parsed);
    out.spec.name = "_weight_offload_internal_budget_bytes";
    out.spec.value = out.budget_bytes;
    out.spec.value_is_mb = false;
    return out;
  }
  if (bytes_res.error() == Error::InvalidArgument) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] "
        "_weight_offload_internal_budget_bytes is set but not a string\n");
    return Error::InvalidArgument;
  }
  if (bytes_res.error() != Error::NotFound) {
    return bytes_res.error();
  }

  // Default: floor + pinned. Checked addition: a corrupt payload
  // with absurd floor / pinned could wrap silently.
  if (floor_bytes > std::numeric_limits<uint64_t>::max() - pinned_bytes_total) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] default budget overflows "
        "(floor=%llu + pinned=%llu > UINT64_MAX)\n",
        static_cast<unsigned long long>(floor_bytes),
        static_cast<unsigned long long>(pinned_bytes_total));
    return Error::Internal;
  }
  out.budget_bytes = floor_bytes + pinned_bytes_total;
  // out.spec.value stays 0 to indicate "default path"; the UX hint
  // still names the public spec as the recommended knob.
  return out;
}

} // namespace

::executorch::runtime::Result<AOTICatalog> walk_aoti_catalog(
    ::executorch::backends::aoti::AOTIDelegateHandle* handle,
    const std::string& method_name) {
  using ::executorch::runtime::Error;

  if (handle->get_num_constants == nullptr ||
      handle->get_constant_name == nullptr ||
      handle->get_constant_original_fqn == nullptr ||
      handle->update_user_managed_constant_buffer_pairs == nullptr ||
      handle->get_constant_data_size == nullptr ||
      handle->get_constant_from_folded == nullptr) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': required AOTI symbols "
        "unresolved (get_num_constants=%p get_constant_name=%p "
        "get_constant_original_fqn=%p update_user_managed=%p "
        "get_constant_data_size=%p get_constant_from_folded=%p)\n",
        method_name.c_str(),
        reinterpret_cast<void*>(handle->get_num_constants),
        reinterpret_cast<void*>(handle->get_constant_name),
        reinterpret_cast<void*>(handle->get_constant_original_fqn),
        reinterpret_cast<void*>(
            handle->update_user_managed_constant_buffer_pairs),
        reinterpret_cast<void*>(handle->get_constant_data_size),
        reinterpret_cast<void*>(handle->get_constant_from_folded));
    return Error::Internal;
  }

  AOTICatalog out;
  if (handle->get_num_constants(handle->container_handle, &out.num_constants) !=
      Error::Ok) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] get_num_constants failed for "
        "method '%s'\n",
        method_name.c_str());
    return Error::Internal;
  }
  out.fqns.reserve(out.num_constants);
  out.internal_names.reserve(out.num_constants);
  out.data_sizes.reserve(out.num_constants);
  out.fqn_to_index.reserve(out.num_constants);

  for (size_t i = 0; i < out.num_constants; ++i) {
    bool folded = false;
    if (handle->get_constant_from_folded(
            handle->container_handle, i, &folded) != Error::Ok) {
      return Error::Internal;
    }
    if (folded) {
      const char* iname = nullptr;
      (void)handle->get_constant_name(handle->container_handle, i, &iname);
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': folded constant at "
          "idx=%zu (internal_name='%s') -- disable "
          "torch._inductor.config.aot_inductor.use_runtime_constant_folding\n",
          method_name.c_str(),
          i,
          iname == nullptr ? "<null>" : iname);
      return Error::InvalidArgument;
    }

    const char* fqn_cstr = nullptr;
    if (handle->get_constant_original_fqn(
            handle->container_handle, i, &fqn_cstr) != Error::Ok ||
        fqn_cstr == nullptr || fqn_cstr[0] == '\0') {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': AOTI constant "
          "idx=%zu has empty/null original_fqn (cannot probe-route)\n",
          method_name.c_str(),
          i);
      return Error::InvalidArgument;
    }
    const char* iname_cstr = nullptr;
    if (handle->get_constant_name(handle->container_handle, i, &iname_cstr) !=
            Error::Ok ||
        iname_cstr == nullptr) {
      return Error::Internal;
    }
    size_t ds = 0;
    if (handle->get_constant_data_size(handle->container_handle, i, &ds) !=
        Error::Ok) {
      return Error::Internal;
    }
    out.fqns.emplace_back(fqn_cstr);
    out.internal_names.emplace_back(iname_cstr);
    out.data_sizes.push_back(ds);
    auto [it, inserted] = out.fqn_to_index.emplace(out.fqns.back(), i);
    if (!inserted) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': duplicate AOTI "
          "original_fqn '%s' at indices %zu and %zu\n",
          method_name.c_str(),
          out.fqns.back().c_str(),
          it->second,
          i);
      return Error::InvalidArgument;
    }
  }
  return out;
}

::executorch::runtime::Result<std::unique_ptr<Session>> Session::create(
    const Payload& payload,
    ::executorch::backends::aoti::AOTIDelegateHandle* handle,
    AOTICatalog catalog,
    const uint8_t* weights_blob,
    cudaStream_t compute_stream,
    ::executorch::runtime::BackendInitContext& context) {
  using ::executorch::runtime::Error;

  // Build fqn -> ConstantMetadata index. The parser validated
  // metadata FQNs == unique(schedule), so every scheduled FQN is
  // present and unique.
  std::unordered_map<std::string, const ConstantMetadata*> fqn_to_meta;
  fqn_to_meta.reserve(payload.constants_metadata.size());
  for (const auto& m : payload.constants_metadata) {
    fqn_to_meta.emplace(m.fqn, &m);
  }

  // Pinned bytes: parser-validated nbytes == elementSize * product,
  // and pin_fqns are deduped + subset of schedule, so no overflow
  // check needed beyond what the parser already did per entry.
  uint64_t pinned_bytes_total = 0;
  for (const auto& fqn : payload.pin_fqns) {
    pinned_bytes_total += fqn_to_meta.at(fqn)->nbytes;
  }

  // Resolve the load-time budget from the runtime-spec chain.
  auto budget_res =
      resolve_budget(context, payload.floor_bytes, pinned_bytes_total);
  if (!budget_res.ok()) {
    return budget_res.error();
  }
  const uint64_t budget_bytes = budget_res.get().budget_bytes;
  const BudgetSpec budget_spec = budget_res.get().spec;

  auto session = std::unique_ptr<Session>(new Session());
  session->schedule_ = payload.schedule;
  session->compute_stream_ = compute_stream;
  session->total_budget_bytes_ = budget_bytes;
  session->pinned_bytes_total_ = pinned_bytes_total;
  session->floor_bytes_ = payload.floor_bytes;
  session->method_name_ = payload.method_name;
  // Single-device today: the parser hard-fails device_index != 0.
  session->device_index_ = 0;

  std::unordered_set<std::string> unique_fqns(
      payload.schedule.begin(), payload.schedule.end());
  session->host_entries_.reserve(unique_fqns.size());
  session->registered_dummies_.reserve(unique_fqns.size());
  for (const auto& fqn : payload.pin_fqns) {
    session->pin_set_.insert(fqn);
  }

  const uint64_t required_total = payload.floor_bytes + pinned_bytes_total;

  // total < pinned + floor covers both "pinned alone exceeds total"
  // and "streaming budget < floor". Suggest the public spec name for
  // the fix (rounded up to whole MB so the value is loadable).
  if (session->total_budget_bytes_ < required_total) {
    constexpr uint64_t kMiB = 1ull << 20;
    const uint64_t required_mb =
        (required_total / kMiB) + ((required_total % kMiB) != 0 ? 1 : 0);
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': budget %llu bytes < "
        "required total %llu (pinned constants %llu + streaming pool "
        "floor %llu)",
        payload.method_name.c_str(),
        static_cast<unsigned long long>(session->total_budget_bytes_),
        static_cast<unsigned long long>(required_total),
        static_cast<unsigned long long>(session->pinned_bytes_total_),
        static_cast<unsigned long long>(payload.floor_bytes));
    if (budget_spec.value > 0) {
      std::fprintf(
          stderr,
          " [set via %s=%llu]",
          budget_spec.name,
          static_cast<unsigned long long>(budget_spec.value));
    }
    std::fprintf(
        stderr,
        ". Set weight_offload_budget_mb >= %llu. "
        "NOTE: floor is a conservative FX-fusion upper bound "
        "(max consecutive pair-window working set + eviction "
        "headroom for the largest single weight) — the actual "
        "runtime peak working set is typically smaller. See "
        "_compute_floor_bytes in backends/cuda/passes/"
        "weight_offload_pass.py for the exact formula.\n",
        static_cast<unsigned long long>(required_mb));
    return Error::InvalidArgument;
  }

  // Past the unified check: total >= pinned + floor, so streaming is
  // non-negative AND >= floor.
  session->streaming_budget_bytes_ =
      session->total_budget_bytes_ - session->pinned_bytes_total_;

  // Bind the device BEFORE any device operation (dummy cudaMalloc,
  // host-mirror H2D, stream/pool create, pinned cudaMalloc). All
  // subsequent CUDA calls implicitly use the current device.
  cudaError_t dev_err = cudaSetDevice(session->device_index_);
  if (dev_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] cudaSetDevice(%d) failed: %s\n",
        session->device_index_,
        cudaGetErrorString(dev_err));
    return Error::Internal;
  }

  // Build dummies + install via AOTI. One 1-byte cudaMalloc + SlimTensor
  // wrap per AOTI constant; any failure inside the loop returns and the
  // unique_ptr<Session>'s dtor runs ~Session() which calls
  // dummies_.clear_and_free() to free the partial state.
  DummyInstallation dummy_guard;
  dummy_guard.dummy_data_ptrs.reserve(catalog.num_constants);
  dummy_guard.dummy_tensors.reserve(catalog.num_constants);
  std::vector<::executorch::backends::aoti::AOTInductorConstantMapEntry> pairs;
  pairs.reserve(catalog.num_constants);
  static constexpr int64_t kDummySizes[1] = {1};
  static constexpr int64_t kDummyStrides[1] = {1};
  // Float (c10::ScalarType::Float == 6). The dummy is opaque — no
  // kernel reads it; the probe op redirects every read.
  static constexpr int32_t kDummyDtype = 6;
  for (size_t i = 0; i < catalog.num_constants; ++i) {
    void* gpu = nullptr;
    if (cudaMalloc(&gpu, 1) != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaMalloc(1) for dummy idx=%zu "
          "failed\n",
          i);
      session->dummies_ = std::move(dummy_guard);
      return Error::Internal;
    }
    dummy_guard.dummy_data_ptrs.push_back(gpu);
    ::executorch::backends::aoti::Tensor* st = nullptr;
    auto wrap_err =
        ::executorch::backends::cuda::aoti_torch_create_tensor_from_blob_v2(
            gpu,
            /*ndim=*/1,
            const_cast<int64_t*>(kDummySizes),
            const_cast<int64_t*>(kDummyStrides),
            /*storage_offset=*/0,
            kDummyDtype,
            /*device_type=*/1,
            /*device_index=*/0,
            &st,
            /*layout=*/0,
            /*opaque_metadata=*/nullptr,
            /*opaque_metadata_size=*/0);
    if (wrap_err != Error::Ok || st == nullptr) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] dummy wrap idx=%zu failed (0x%x)\n",
          i,
          static_cast<uint32_t>(wrap_err));
      session->dummies_ = std::move(dummy_guard);
      return Error::Internal;
    }
    dummy_guard.dummy_tensors.push_back(st);
    pairs.push_back(
        {catalog.internal_names[i].c_str(),
         reinterpret_cast<::executorch::backends::aoti::AtenTensorHandle>(st)});
  }
  // Install dummies via AOTI. After this, AOTI dispatches probes
  // with the dummy SlimTensor as input; ProbeRegistry routes back
  // here. validate_full_update=true is belt-and-braces; the caller's
  // catalog/schedule coverage check is the real proof.
  if (handle->update_user_managed_constant_buffer_pairs(
          handle->container_handle,
          pairs.data(),
          pairs.size(),
          /*use_inactive=*/false,
          /*validate_full_update=*/true) != Error::Ok) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] method '%s': "
        "update_user_managed_constant_buffer_pairs failed\n",
        payload.method_name.c_str());
    session->dummies_ = std::move(dummy_guard);
    return Error::Internal;
  }
  // Transfer dummy ownership to the session; failures below clean
  // up via the session dtor.
  session->dummies_ = std::move(dummy_guard);

  // Per-FQN source-blob offsets. The blob is densely packed by
  // data_size in declaration order (model_base.h::load_constants's
  // bytes_read += data_size, no alignment padding).
  std::unordered_map<std::string, uint64_t> fqn_offsets;
  fqn_offsets.reserve(catalog.num_constants);
  {
    uint64_t running = 0;
    for (size_t i = 0; i < catalog.num_constants; ++i) {
      fqn_offsets[catalog.fqns[i]] = running;
      running += static_cast<uint64_t>(catalog.data_sizes[i]);
    }
  }

  // Build the pinned host mirror. Bytes are copied DIRECTLY from
  // the caller-provided _weights_blob region (NamedData entry); the
  // caller may free the blob the moment Session::create returns.
  if (weights_blob == nullptr && !unique_fqns.empty()) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] weights_blob is null but schedule is "
        "non-empty\n");
    return Error::InvalidArgument;
  }
  for (const auto& fqn : unique_fqns) {
    const ConstantMetadata& m = *fqn_to_meta.at(fqn);
    const uint64_t blob_offset = fqn_offsets.at(fqn);

    HostEntry entry;
    entry.nbytes = m.nbytes;
    entry.dtype = m.dtype;
    entry.device_index = m.device_index;
    entry.sizes = m.sizes;
    entry.strides = m.strides;

    cudaError_t alloc_err =
        cudaHostAlloc(&entry.host_ptr, entry.nbytes, cudaHostAllocDefault);
    if (alloc_err != cudaSuccess || entry.host_ptr == nullptr) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaHostAlloc(%llu) for FQN '%s' "
          "failed: %s\n",
          static_cast<unsigned long long>(entry.nbytes),
          fqn.c_str(),
          cudaGetErrorString(alloc_err));
      return Error::Internal;
    }
    std::memcpy(entry.host_ptr, weights_blob + blob_offset, entry.nbytes);
    session->host_entries_.emplace(fqn, std::move(entry));
  }

  // 6. Create the copy stream (separate from compute_stream_).
  cudaError_t stream_err =
      cudaStreamCreateWithFlags(&session->copy_stream_, cudaStreamNonBlocking);
  if (stream_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] cudaStreamCreateWithFlags failed: %s\n",
        cudaGetErrorString(stream_err));
    return Error::Internal;
  }

  // 7. Create the pool with PyTorch-style reuse flags. The set-
  //    attribute calls are NOT optional — the cross-stream reuse
  //    semantics the hot path relies on require these flags. A
  //    failure here means the pool would not behave as designed,
  //    so hard-fail rather than silently degrade.
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = session->device_index_;
  cudaError_t pool_err = cudaMemPoolCreate(&session->pool_, &props);
  if (pool_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] cudaMemPoolCreate failed: %s\n",
        cudaGetErrorString(pool_err));
    return Error::Internal;
  }
  int yes = 1;
  struct PoolAttr {
    cudaMemPoolAttr attr;
    void* value;
    const char* name;
  };
  // Release threshold is the pool's preferred upper bound on
  // driver-side cache. It is SOFT (PyTorch's comment) — the pool
  // CAN reserve more if pressure demands it — so this is not a
  // hard cap on physical footprint. Setting it to
  // streaming_budget_bytes_ (rather than total) keeps the
  // preferred bound aligned with our REQUESTED-live-bytes
  // invariant: bytes_in_flight_ + pinned_bytes_total_ <=
  // total_budget_bytes_. Driver-reserved / cached memory may
  // briefly exceed pinned + streaming, but our accounting
  // (peak_live_bytes <= streaming_budget) stays self-consistent.
  uint64_t release_threshold = session->streaming_budget_bytes_;
  const PoolAttr pool_attrs[] = {
      {cudaMemPoolReuseFollowEventDependencies,
       &yes,
       "cudaMemPoolReuseFollowEventDependencies"},
      {cudaMemPoolReuseAllowOpportunistic,
       &yes,
       "cudaMemPoolReuseAllowOpportunistic"},
      {cudaMemPoolReuseAllowInternalDependencies,
       &yes,
       "cudaMemPoolReuseAllowInternalDependencies"},
      {cudaMemPoolAttrReleaseThreshold,
       &release_threshold,
       "cudaMemPoolAttrReleaseThreshold"},
  };
  for (const auto& pa : pool_attrs) {
    cudaError_t set_err =
        cudaMemPoolSetAttribute(session->pool_, pa.attr, pa.value);
    if (set_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaMemPoolSetAttribute(%s) failed: "
          "%s\n",
          pa.name,
          cudaGetErrorString(set_err));
      return Error::Internal;
    }
  }

  // 7b. Allocate pinned constants. One ``cudaMalloc`` per pinned FQN,
  //     ``cudaMemcpyAsync`` from the pinned host mirror on
  //     ``compute_stream_``, then ``cudaStreamSynchronize`` so
  //     Session::serve's pinned fast path can return without any
  //     event wait. ``pinned_.emplace`` asserts the parse-time dedupe
  //     held — duplicate FQN here would shadow the prior pointer
  //     and leak the GPU buffer.
  //
  //     Pinned cudaMalloc bypasses the offload pool: pinned
  //     allocations are out-of-band and don't count toward
  //     ``bytes_in_flight_`` against ``streaming_budget_bytes_``.
  //     They're cleaned up in ``~Session()`` between
  //     ``dummies_.clear_and_free()`` and ``free_host_mirror``.
  if (!session->pin_set_.empty()) {
    // Defensive re-bind: AOTI's update_user_managed_constant_buffer_pairs
    // contract doesn't promise the current device is preserved (it
    // doesn't internally re-bind today, but the contract is silent).
    // Pinned cudaMalloc lands wherever the current device points, so
    // re-bind here to keep pinned allocations on device_index_ even
    // if AOTI ever flips the current device.
    cudaError_t pin_dev_err = cudaSetDevice(session->device_index_);
    if (pin_dev_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaSetDevice(%d) before pinned "
          "alloc failed: %s\n",
          session->device_index_,
          cudaGetErrorString(pin_dev_err));
      return Error::Internal;
    }
    bool pin_alloc_ok = true;
    for (const auto& fqn : session->pin_set_) {
      const HostEntry& host = session->host_entries_.at(fqn);
      void* dev = nullptr;
      cudaError_t pin_alloc_err = cudaMalloc(&dev, host.nbytes);
      if (pin_alloc_err != cudaSuccess) {
        std::fprintf(
            stderr,
            "[ET_WEIGHT_OFFLOAD][ERROR] cudaMalloc(%llu) for pinned "
            "FQN '%s' failed: %s\n",
            static_cast<unsigned long long>(host.nbytes),
            fqn.c_str(),
            cudaGetErrorString(pin_alloc_err));
        pin_alloc_ok = false;
        break;
      }
      cudaError_t copy_err = cudaMemcpyAsync(
          dev,
          host.host_ptr,
          host.nbytes,
          cudaMemcpyHostToDevice,
          session->compute_stream_);
      if (copy_err != cudaSuccess) {
        std::fprintf(
            stderr,
            "[ET_WEIGHT_OFFLOAD][ERROR] cudaMemcpyAsync H2D for pinned "
            "FQN '%s' (%llu bytes) failed: %s\n",
            fqn.c_str(),
            static_cast<unsigned long long>(host.nbytes),
            cudaGetErrorString(copy_err));
        cudaFree(dev);
        pin_alloc_ok = false;
        break;
      }
      auto [it, inserted] = session->pinned_.emplace(fqn, dev);
      (void)it;
      if (!inserted) {
        std::fprintf(
            stderr,
            "[ET_WEIGHT_OFFLOAD][ERROR] duplicate pinned FQN '%s' "
            "during alloc — parse-time dedupe was bypassed\n",
            fqn.c_str());
        cudaFree(dev);
        pin_alloc_ok = false;
        break;
      }
    }
    if (!pin_alloc_ok) {
      // pinned_ accumulated partial state; let unique_ptr<Session>'s
      // dtor clean it up (the dtor's pinned-cleanup loop runs
      // unconditionally over pinned_ entries).
      return Error::Internal;
    }
    // Drain the pinned H2D before returning so the fast path can
    // serve pinned constants without an event wait.
    cudaError_t sync_err = cudaStreamSynchronize(session->compute_stream_);
    if (sync_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaStreamSynchronize after pinned "
          "H2D failed: %s\n",
          cudaGetErrorString(sync_err));
      return Error::Internal;
    }
  }

  // Register dummies with ProbeRegistry -- LAST step. Each AOTI
  // constant index has a unique 1-byte cudaMalloc'd dummy (built
  // above), so the data_ptrs are already unique; just hand the
  // vector to the registry.
  std::vector<const void*> dummy_keys(
      session->dummies_.dummy_data_ptrs.begin(),
      session->dummies_.dummy_data_ptrs.end());
  auto& registry = ProbeRegistry::instance();
  auto reg_err = registry.register_entries(
      session.get(),
      &Session::registry_callback,
      dummy_keys.data(),
      dummy_keys.size());
  if (reg_err != Error::Ok) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] ProbeRegistry::register_entries failed "
        "(error 0x%x); collision against an already-registered pointer is "
        "the only path that returns here\n",
        static_cast<uint32_t>(reg_err));
    return reg_err;
  }
  session->registered_dummies_ = std::move(dummy_keys);

  return session;
}

Session::~Session() {
  // Unregister from ProbeRegistry FIRST — pure CPU work, safe even
  // if the GPU is in a bad state. After this point no new probe
  // calls can land on serve().
  ProbeRegistry::instance().unregister_context(
      this, registered_dummies_.data(), registered_dummies_.size());

  // Re-bind the device for GPU teardown. cudaStreamSynchronize /
  // cudaFreeAsync / cudaEventDestroy are stream-or-pointer-bound
  // and don't strictly require the current device match, but
  // cudaMemPoolDestroy and cudaFree (for dummies / pinned) target
  // resources that live on device_index_, and operating on them
  // with a wrong current device risks corrupting allocations on
  // whatever device IS current. If the re-bind fails, SKIP all GPU
  // cleanup — the GPU-side resources leak (unrecoverable in this
  // error path), but we don't double-fault into another device's
  // state. We still drop the CPU-side bookkeeping (live_/lru_order_/
  // pinned_/dummies_/host_mirror) so the Session object itself is
  // safe to destroy.
  cudaError_t set_dev_err = cudaSetDevice(device_index_);
  if (set_dev_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][WARN] cudaSetDevice(%d) in ~Session() "
        "failed: %s; SKIPPING GPU teardown to avoid touching the "
        "wrong device. Streams/pool/dummies/pinned GPU buffers "
        "will leak.\n",
        device_index_,
        cudaGetErrorString(set_dev_err));
    // Drop bookkeeping that owns GPU handles WITHOUT touching the
    // handles themselves. SlimTensor wrappers from
    // aoti_torch_create_tensor_from_blob_v2 are non-owning
    // (``from_blob``) so ``delete`` on them is pure CPU work —
    // safe even with a bad GPU state. The dummy cudaMalloc'd
    // device pointers DO leak; clearing the ptr vector first
    // ensures the implicit dummies_ dtor doesn't try to cudaFree
    // them.
    for (auto* t : dummies_.dummy_tensors) {
      delete t;
    }
    dummies_.dummy_tensors.clear();
    dummies_.dummy_data_ptrs.clear();
    live_.clear();
    lru_order_.clear();
    pinned_.clear();
    host_entries_.clear();
    return;
  }

  if (compute_stream_ != nullptr) {
    // Drain kernels that may be reading any live_ entry.
    cudaStreamSynchronize(compute_stream_);
  }

  for (auto& [fqn, e] : live_) {
    if (e.dev_ptr != nullptr && compute_stream_ != nullptr) {
      // Stream-order the free behind the entry's H2D. Prefetched
      // entries may have a ready_event no prior serve() has waited
      // on; without this wait, cudaFreeAsync on compute_stream_
      // would queue the free before the in-flight cudaMemcpyAsync
      // on copy_stream_ drains and free memory still being written
      // to. cudaStreamWaitEvent is a queued cross-stream dependency
      // (not a host sync), and a no-op if the event has already
      // completed.
      if (e.ready_event != nullptr) {
        cudaStreamWaitEvent(compute_stream_, e.ready_event, 0);
      }
      cudaFreeAsync(e.dev_ptr, compute_stream_);
    }
    if (e.ready_event != nullptr) {
      cudaEventDestroy(e.ready_event);
    }
  }
  live_.clear();
  lru_order_.clear();
  bytes_in_flight_ = 0;

  if (compute_stream_ != nullptr) {
    // Make the async frees enqueued above actually take effect
    // before destroying the pool.
    cudaStreamSynchronize(compute_stream_);
  }
  if (copy_stream_ != nullptr) {
    // Drain any work the serve() free_on_error paths left on
    // copy_stream_; compute_stream_'s sync doesn't cover them.
    cudaStreamSynchronize(copy_stream_);
    cudaStreamDestroy(copy_stream_);
    copy_stream_ = nullptr;
  }
  if (pool_ != nullptr) {
    cudaMemPoolDestroy(pool_);
    pool_ = nullptr;
  }

  // Free the dummies BETWEEN pool destroy and host-mirror free. By
  // this point all streams + the pool are torn down, so no in-flight
  // GPU work can touch the 1-byte dummy regions. AOTI's
  // ``constants_map_`` is left pointing at the freed handles —
  // harmless because run() is never invoked again on this container
  // and the container itself is never explicitly deleted.
  dummies_.clear_and_free();

  // Free pinned GPU allocations. Same ordering as dummies — after
  // pool/stream teardown so no in-flight GPU work touches the
  // pinned regions, before host_mirror so the pinned host buffers
  // we copied from are still alive (not strictly required for
  // cudaFree, but matches the dummies' ordering for consistency).
  for (auto& [fqn, dev] : pinned_) {
    if (dev != nullptr) {
      cudaFree(dev);
    }
  }
  pinned_.clear();

  free_host_mirror(host_entries_);

  if (std::getenv("EXECUTORCH_WEIGHT_OFFLOAD_TRACE") != nullptr) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD_STATS] method=%s hits=%llu misses=%llu "
        "evictions=%llu bytes_h2d=%llu peak_live_bytes=%llu budget=%llu "
        "floor=%llu prefetch_attempted=%llu prefetch_succeeded=%llu "
        "pinned_bytes=%llu streaming_budget=%llu\n",
        method_name_.c_str(),
        static_cast<unsigned long long>(stats_.pool_hits),
        static_cast<unsigned long long>(stats_.pool_misses),
        static_cast<unsigned long long>(stats_.evictions),
        static_cast<unsigned long long>(stats_.bytes_h2d_copied),
        static_cast<unsigned long long>(peak_live_bytes_),
        static_cast<unsigned long long>(total_budget_bytes_),
        static_cast<unsigned long long>(floor_bytes_),
        static_cast<unsigned long long>(stats_.prefetch_attempted),
        static_cast<unsigned long long>(stats_.prefetch_succeeded),
        static_cast<unsigned long long>(pinned_bytes_total_),
        static_cast<unsigned long long>(streaming_budget_bytes_));
  }
}

::executorch::runtime::Error Session::registry_callback(
    void* context,
    ::executorch::backends::aoti::Tensor* input,
    int64_t probe_id,
    ::executorch::backends::aoti::Tensor** output) {
  return static_cast<Session*>(context)->serve(input, probe_id, output);
}

std::unordered_map<std::string, Session::LiveAllocation>::iterator
Session::pick_lru() {
  // O(1): the front of lru_order_ is the least-recently-used FQN.
  // Pinned entries never enter live_/lru_order_ (they live in
  // pinned_, allocated out-of-pool via cudaMalloc).
  if (lru_order_.empty()) {
    return live_.end();
  }
  return live_.find(lru_order_.front());
}

::executorch::runtime::Error Session::wrap_borrowed_tensor(
    void* dev_ptr,
    const HostEntry& host,
    ::executorch::backends::aoti::Tensor** output) {
  using ::executorch::runtime::Error;
  ::executorch::backends::aoti::Tensor* wrapped = nullptr;
  auto err =
      ::executorch::backends::cuda::aoti_torch_create_tensor_from_blob_v2(
          dev_ptr,
          static_cast<int64_t>(host.sizes.size()),
          host.sizes.data(),
          host.strides.data(),
          /*storage_offset=*/0,
          host.dtype,
          kCudaDeviceType,
          host.device_index,
          &wrapped,
          /*layout=*/0,
          /*opaque_metadata=*/nullptr,
          /*opaque_metadata_size=*/0);
  if (err != Error::Ok || wrapped == nullptr) {
    return err == Error::Ok ? Error::Internal : err;
  }
  *output = wrapped;
  return Error::Ok;
}

::executorch::runtime::Error Session::make_room_and_alloc(
    const std::string& fqn,
    const HostEntry& host,
    const std::string* guard_fqn,
    const char* log_tag,
    void** dev_out,
    cudaEvent_t* ready_out) {
  using ::executorch::runtime::Error;
  *dev_out = nullptr;
  *ready_out = nullptr;
  const uint64_t need = host.nbytes;

  // 1. Eviction loop. Stream-wait on each victim's ready_event before
  //    cudaFreeAsync so a prefetched-but-not-yet-consumed entry isn't
  //    freed mid-H2D.
  bool evicted = false;
  while (bytes_in_flight_ + need > streaming_budget_bytes_) {
    auto victim_it = pick_lru();
    if (victim_it == live_.end()) {
      std::fprintf(
          stderr,
          "%s cannot make room for FQN '%s' needing %llu bytes; "
          "streaming_budget=%llu, bytes_in_flight=%llu\n",
          log_tag,
          fqn.c_str(),
          static_cast<unsigned long long>(need),
          static_cast<unsigned long long>(streaming_budget_bytes_),
          static_cast<unsigned long long>(bytes_in_flight_));
      return Error::Internal;
    }
    if (guard_fqn != nullptr && victim_it->first == *guard_fqn) {
      // Prefetch's "never evict the just-served FQN" guard. The
      // floor invariant should prevent this; if it ever fires, the
      // caller should skip the prefetch instead of risking
      // corruption of the in-flight entry.
      std::fprintf(
          stderr,
          "%s LRU victim for FQN '%s' would be the guarded '%s'; "
          "floor invariant may be violated\n",
          log_tag,
          fqn.c_str(),
          guard_fqn->c_str());
      return Error::Internal;
    }
    auto& v = victim_it->second;
    cudaError_t wait_err =
        cudaStreamWaitEvent(compute_stream_, v.ready_event, 0);
    if (wait_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "%s cudaStreamWaitEvent on victim '%s': %s\n",
          log_tag,
          victim_it->first.c_str(),
          cudaGetErrorString(wait_err));
      return Error::Internal;
    }
    cudaError_t free_err = cudaFreeAsync(v.dev_ptr, compute_stream_);
    if (free_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "%s cudaFreeAsync for evicted FQN '%s' (%llu bytes): %s; "
          "live_ + bytes_in_flight_ left untouched\n",
          log_tag,
          victim_it->first.c_str(),
          static_cast<unsigned long long>(v.nbytes),
          cudaGetErrorString(free_err));
      return Error::Internal;
    }
    cudaEventDestroy(v.ready_event);
    bytes_in_flight_ -= v.nbytes;
    stats_.evictions++;
    lru_order_.erase(v.lru_it);
    live_.erase(victim_it);
    evicted = true;
  }

  // 2. After eviction batch: record event on compute_, copy_ waits on
  //    it before the upcoming malloc. Failure here is delicate
  //    (live_/bytes_in_flight_ already mutated); fall back to a host
  //    sync on compute_ so the frees physically complete and Session
  //    state stays consistent.
  if (evicted) {
    cudaEvent_t evict_done = nullptr;
    cudaError_t ev_err =
        cudaEventCreateWithFlags(&evict_done, cudaEventDisableTiming);
    if (ev_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "%s cudaEventCreate for eviction batch: %s; syncing compute_stream\n",
          log_tag,
          cudaGetErrorString(ev_err));
      (void)cudaStreamSynchronize(compute_stream_);
      return Error::Internal;
    }
    if (cudaEventRecord(evict_done, compute_stream_) != cudaSuccess ||
        cudaStreamWaitEvent(copy_stream_, evict_done, 0) != cudaSuccess) {
      cudaEventDestroy(evict_done);
      std::fprintf(
          stderr,
          "%s event-record/wait for eviction batch failed; syncing "
          "compute_stream\n",
          log_tag);
      (void)cudaStreamSynchronize(compute_stream_);
      return Error::Internal;
    }
    cudaEventDestroy(evict_done);
  }

  // 3. Allocate on copy_. From here on, any failure must
  //    cudaFreeAsync(dev) before returning.
  void* dev = nullptr;
  cudaError_t malloc_err =
      cudaMallocFromPoolAsync(&dev, need, pool_, copy_stream_);
  if (malloc_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "%s cudaMallocFromPoolAsync for FQN '%s' (%llu bytes): %s\n",
        log_tag,
        fqn.c_str(),
        static_cast<unsigned long long>(need),
        cudaGetErrorString(malloc_err));
    return Error::Internal;
  }
  auto free_on_error = [&]() {
    if (dev != nullptr) {
      cudaFreeAsync(dev, copy_stream_);
      dev = nullptr;
    }
  };

  cudaError_t memcpy_err = cudaMemcpyAsync(
      dev, host.host_ptr, need, cudaMemcpyHostToDevice, copy_stream_);
  if (memcpy_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "%s cudaMemcpyAsync H2D for FQN '%s' (%llu bytes): %s\n",
        log_tag,
        fqn.c_str(),
        static_cast<unsigned long long>(need),
        cudaGetErrorString(memcpy_err));
    free_on_error();
    return Error::Internal;
  }

  cudaEvent_t ready = nullptr;
  cudaError_t ev_err = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
  if (ev_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "%s cudaEventCreate ready for FQN '%s': %s\n",
        log_tag,
        fqn.c_str(),
        cudaGetErrorString(ev_err));
    free_on_error();
    return Error::Internal;
  }
  if (cudaEventRecord(ready, copy_stream_) != cudaSuccess) {
    cudaEventDestroy(ready);
    free_on_error();
    std::fprintf(
        stderr,
        "%s cudaEventRecord ready for FQN '%s'\n",
        log_tag,
        fqn.c_str());
    return Error::Internal;
  }

  *dev_out = dev;
  *ready_out = ready;
  return Error::Ok;
}

::executorch::runtime::Error Session::serve(
    ::executorch::backends::aoti::Tensor* input,
    int64_t probe_id,
    ::executorch::backends::aoti::Tensor** output) {
  using ::executorch::runtime::Error;
  (void)input; // identity comes from schedule_[probe_id]

  // Re-bind the device only if something external (a Python extension,
  // another backend, a sibling thread) flipped the current device
  // since the cudaSetDevice in CudaBackend::execute(). The pool,
  // copy_stream_, and dummy allocations all live on device_index_;
  // cudaMallocFromPoolAsync / cudaEventCreate /
  // aoti_torch_create_tensor_from_blob_v2 all implicitly use the
  // current device, so a mismatch would corrupt across GPUs.
  //
  // cudaGetDevice is a much cheaper call than cudaSetDevice; checking
  // first turns the steady-state cost into one get per probe instead
  // of one set, which matters for models with hundreds of probes per
  // forward pass.
  int current_device = -1;
  cudaError_t get_dev_err = cudaGetDevice(&current_device);
  if (get_dev_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] cudaGetDevice in serve() failed: %s\n",
        cudaGetErrorString(get_dev_err));
    return Error::Internal;
  }
  if (current_device != device_index_) {
    cudaError_t set_dev_err = cudaSetDevice(device_index_);
    if (set_dev_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaSetDevice(%d) in serve() failed: "
          "%s\n",
          device_index_,
          cudaGetErrorString(set_dev_err));
      return Error::Internal;
    }
  }

  if (probe_id < 0 || static_cast<size_t>(probe_id) >= schedule_.size()) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] probe_id=%" PRId64
        " out of range for "
        "schedule of size %zu\n",
        probe_id,
        schedule_.size());
    return Error::InvalidArgument;
  }
  const std::string& fqn = schedule_[probe_id];
  auto host_it = host_entries_.find(fqn);
  if (host_it == host_entries_.end()) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] no host mirror for FQN '%s' "
        "(probe_id=%" PRId64 ")\n",
        fqn.c_str(),
        probe_id);
    return Error::Internal;
  }
  const HostEntry& host = host_it->second;

  // Pinned fast path. Resident GPU allocation, populated once at
  // init via cudaMemcpyAsync + sync; no event wait, no pool work,
  // no streaming-stats bump. STILL calls opportunistic_prefetch
  // at the end so a pinned→streaming transition doesn't lose
  // overlap.
  if (auto pin_it = pinned_.find(fqn); pin_it != pinned_.end()) {
    ::executorch::backends::aoti::Tensor* wrapped = nullptr;
    auto wrap_err = wrap_borrowed_tensor(pin_it->second, host, &wrapped);
    if (wrap_err != Error::Ok) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] pinned-path borrowed-wrap for FQN "
          "'%s' failed (error 0x%x)\n",
          fqn.c_str(),
          static_cast<uint32_t>(wrap_err));
      return wrap_err;
    }
    *output = wrapped;
    (void)opportunistic_prefetch(probe_id);
    return Error::Ok;
  }

  // Pool hit.
  if (auto live_it = live_.find(fqn); live_it != live_.end()) {
    auto& e = live_it->second;
    cudaError_t wait_err =
        cudaStreamWaitEvent(compute_stream_, e.ready_event, 0);
    if (wait_err != cudaSuccess) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] cudaStreamWaitEvent on hit for FQN "
          "'%s' failed: %s\n",
          fqn.c_str(),
          cudaGetErrorString(wait_err));
      return Error::Internal;
    }
    // Splice to back of lru_order_: O(1) and preserves the
    // iterator, so e.lru_it stays valid.
    lru_order_.splice(lru_order_.end(), lru_order_, e.lru_it);
    stats_.pool_hits++;
    ::executorch::backends::aoti::Tensor* wrapped = nullptr;
    auto err = wrap_borrowed_tensor(e.dev_ptr, host, &wrapped);
    if (err != Error::Ok) {
      std::fprintf(
          stderr,
          "[ET_WEIGHT_OFFLOAD][ERROR] hit-path borrowed-wrap for FQN '%s' "
          "failed (error 0x%x)\n",
          fqn.c_str(),
          static_cast<uint32_t>(err));
      return err;
    }
    *output = wrapped;
    // Best-effort depth-1 prefetch. Errors are logged inside the
    // helper and never propagated — the current probe is already
    // populated in *output.
    (void)opportunistic_prefetch(probe_id);
    return Error::Ok;
  }

  // Miss.
  stats_.pool_misses++;
  const uint64_t need = host.nbytes;

  void* dev = nullptr;
  cudaEvent_t ready = nullptr;
  auto room_err = make_room_and_alloc(
      fqn,
      host,
      /*guard_fqn=*/nullptr,
      "[ET_WEIGHT_OFFLOAD][ERROR]",
      &dev,
      &ready);
  if (room_err != Error::Ok) {
    return room_err;
  }

  // Serve-only: gate compute on the H2D ready event.
  cudaError_t wait_err = cudaStreamWaitEvent(compute_stream_, ready, 0);
  if (wait_err != cudaSuccess) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] cudaStreamWaitEvent for FQN '%s': %s\n",
        fqn.c_str(),
        cudaGetErrorString(wait_err));
    cudaEventDestroy(ready);
    cudaFreeAsync(dev, copy_stream_);
    return Error::Internal;
  }

  ::executorch::backends::aoti::Tensor* wrapped = nullptr;
  auto wrap_err = wrap_borrowed_tensor(dev, host, &wrapped);
  if (wrap_err != Error::Ok) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][ERROR] miss-path borrowed-wrap for FQN '%s' "
        "failed (error 0x%x)\n",
        fqn.c_str(),
        static_cast<uint32_t>(wrap_err));
    cudaEventDestroy(ready);
    cudaFreeAsync(dev, copy_stream_);
    return wrap_err;
  }

  bytes_in_flight_ += need;
  peak_live_bytes_ = std::max(peak_live_bytes_, bytes_in_flight_);
  auto lru_it = lru_order_.insert(lru_order_.end(), fqn);
  live_.emplace(fqn, LiveAllocation{dev, need, ready, lru_it});
  stats_.bytes_h2d_copied += need;

  *output = wrapped;
  // Best-effort depth-1 prefetch. Errors are logged inside the
  // helper and never propagated — the current probe is already
  // populated in *output.
  (void)opportunistic_prefetch(probe_id);
  return Error::Ok;
}

::executorch::runtime::Error Session::opportunistic_prefetch(
    int64_t current_probe_id) {
  using ::executorch::runtime::Error;

  if (schedule_.empty()) {
    return Error::Ok;
  }
  const int64_t next_id =
      (current_probe_id + 1) % static_cast<int64_t>(schedule_.size());
  const std::string& fqn = schedule_[static_cast<size_t>(next_id)];

  // Step 1: pinned → already resident, no prefetch needed
  // (cheaper than the live_ check so do it first).
  if (pin_set_.find(fqn) != pin_set_.end()) {
    return Error::Ok;
  }

  // Step 1a: already-live → no work needed.
  if (live_.find(fqn) != live_.end()) {
    return Error::Ok;
  }

  // Defensive guard — never evict the FQN the c-shim is about to
  // hand back to AOTI for kernel launch. The floor formula
  // (budget >= bytes(current) + bytes(next)) should make this
  // unreachable today; the guard covers a hypothetical
  // below-floor configuration for the single-immediately-just-
  // served-FQN case. It does NOT cover multi-probe-before-one-
  // launch (fused kernels with probes A, B before one launch
  // could still have A evicted by a prefetch after B if the floor
  // invariant were violated). The init-time floor hard-fail is
  // the real general contract.
  const std::string& current_fqn =
      schedule_[static_cast<size_t>(current_probe_id)];

  auto host_it = host_entries_.find(fqn);
  if (host_it == host_entries_.end()) {
    std::fprintf(
        stderr,
        "[ET_WEIGHT_OFFLOAD][WARN] prefetch skipped: no host mirror "
        "for FQN '%s'\n",
        fqn.c_str());
    return Error::Internal;
  }
  const HostEntry& host = host_it->second;
  const uint64_t need = host.nbytes;

  // From here on a real prefetch is attempted. Count the attempt
  // regardless of whether it succeeds — `attempted - succeeded`
  // = swallowed errors.
  stats_.prefetch_attempted++;

  void* dev = nullptr;
  cudaEvent_t ready = nullptr;
  auto room_err = make_room_and_alloc(
      fqn,
      host,
      &current_fqn,
      "[ET_WEIGHT_OFFLOAD][WARN] prefetch",
      &dev,
      &ready);
  if (room_err != Error::Ok) {
    return room_err;
  }

  // No cudaStreamWaitEvent(compute_, ready) here — the next serve()
  // that consumes this entry as a hit does that wait itself.
  bytes_in_flight_ += need;
  peak_live_bytes_ = std::max(peak_live_bytes_, bytes_in_flight_);
  // Treat the prefetched entry as "newest": the FQN is about to be
  // served next, so its expected next-use is sooner than what we
  // just served, and a same-cycle miss should evict the older one.
  auto lru_it = lru_order_.insert(lru_order_.end(), fqn);
  live_.emplace(fqn, LiveAllocation{dev, need, ready, lru_it});
  stats_.bytes_h2d_copied += need;
  stats_.prefetch_succeeded++;
  return Error::Ok;
}

} // namespace executorch::backends::cuda::weight_offload

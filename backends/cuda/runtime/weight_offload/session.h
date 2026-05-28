/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// EXPERIMENTAL: per-handle CUDA weight-offload Session.
//
// What it owns: a pinned host mirror of the schedule's constants, a
// bounded cudaMemPool sized by a software byte cap on requested live
// offload bytes, LRU eviction with event-ordered cudaFreeAsync on the
// compute stream, depth-1 opportunistic prefetch on the copy stream,
// optional pinned-resident constants (out-of-pool cudaMalloc), and the
// DummyInstallation that AOTI's container uses as its "active
// constants" so probe ops route every constant read here. See
// session.cpp for the per-step ordering. Single-device only (the
// payload parser hard-fails device_index != 0).
//
// Companion components: payload.{h} (on-wire schema parser, the single
// trust boundary), probe_op.{h,cpp} (AOTI c-shim), probe_registry.{h,cpp}
// (process-global dummy_ptr -> Session map). Public entry from the
// pass side: ``CudaPartitioner(weight_offload=True,
// weight_offload_pin_fqns=[...])``.

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/cuda/runtime/weight_offload/payload.h>
#include <executorch/runtime/backend/backend_init_context.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

namespace executorch::backends::cuda::weight_offload {

// Snapshot of the AOTI container's constant catalog, produced by
// walk_aoti_catalog. Indexed by AOTI constant index. The walk hard-
// fails on folded constants, empty original_fqns, and duplicate
// original_fqns (each of which would silently corrupt the offload
// install).
struct AOTICatalog {
  size_t num_constants{0};
  std::vector<std::string> fqns; // original_fqn by index
  std::vector<std::string> internal_names; // get_constant_name by index
  std::vector<size_t> data_sizes; // get_constant_data_size by index
  std::unordered_map<std::string, size_t> fqn_to_index;
};

// Walk the AOTI container's constants via the function pointers on
// ``handle``. Hard-fails on folded constants, empty original_fqns,
// and duplicate original_fqns. Also verifies the required AOTI
// symbol set is present on the handle.
::executorch::runtime::Result<AOTICatalog> walk_aoti_catalog(
    ::executorch::backends::aoti::AOTIDelegateHandle* handle,
    const std::string& method_name);

// Tag for the below-floor UX hint in Session::create. Populated by
// cuda_backend.cpp from the runtime-spec resolution chain; the
// default-budget path leaves value=0 and sets name to the public
// spec as a forward-looking hint.
struct BudgetSpec {
  const char* name{nullptr};
  uint64_t value{0};
  bool value_is_mb{false}; // false = bytes (internal spec)
};

struct SessionStats {
  uint64_t pool_hits{0};
  uint64_t pool_misses{0};
  uint64_t evictions{0};
  uint64_t bytes_h2d_copied{0};
  // Best-effort depth-1 prefetch counters. Incremented by
  // Session::opportunistic_prefetch (called at the end of every
  // serve). prefetch_attempted bumps BEFORE the H2D is issued;
  // prefetch_succeeded bumps AFTER cudaMemcpyAsync is queued.
  // ``attempted - succeeded`` = swallowed errors (eviction / alloc /
  // copy failure — all logged but never propagated, since the
  // current probe has already returned).
  uint64_t prefetch_attempted{0};
  uint64_t prefetch_succeeded{0};
};

// Owns the 1-byte GPU dummies + SlimTensor wrappers installed via
// AOTI's update_user_managed_constant_buffer_pairs. AOTI's contract
// says the ConstantMap is user-managed; we retain ownership for the
// lifetime of the container. Move-only so a local RAII guard in
// CudaBackend::init can be moved into Session::create at the end of
// the install dance; pre-move failure paths are cleaned up by the
// guard's own dtor.
struct DummyInstallation {
  // Indexed by AOTI constant index (the step-3 hard-fail on folded
  // constants guarantees num_folded == 0, so non-folded position ==
  // AOTI index here).
  std::vector<void*> dummy_data_ptrs; // cudaMalloc'd, 1 byte each.
  std::vector<::executorch::backends::aoti::Tensor*> dummy_tensors;

  DummyInstallation() = default;
  ~DummyInstallation();

  DummyInstallation(DummyInstallation&&) noexcept;
  DummyInstallation& operator=(DummyInstallation&&) noexcept;
  DummyInstallation(const DummyInstallation&) = delete;
  DummyInstallation& operator=(const DummyInstallation&) = delete;

  // Idempotent — Session::~Session calls this explicitly at the right
  // ordering point (after pool/stream teardown, before the field's own
  // dtor runs). The field's dtor then sees empty vectors and does
  // nothing.
  void clear_and_free();
};

class Session {
 public:
  // Build dummies, install them via AOTI, build the host mirror,
  // allocate the pool + copy stream, register dispatch entries with
  // ProbeRegistry. The caller has already parsed + validated the
  // payload and walked the AOTI catalog (folded check + coverage +
  // AOTI/payload data_size cross-check); Session::create trusts both
  // inputs and owns everything from dummy install through registry
  // registration.
  //
  // ``weights_blob`` is a borrowed pointer into the NamedDataMap's
  // ``<method>_weights_blob`` entry; Session::create copies the
  // per-FQN bytes into pinned host memory and the caller may free
  // ``weights_blob`` the moment create returns.
  //
  // Resolves the load-time budget from the runtime-spec chain
  // (public ``weight_offload_budget_mb`` -> internal
  // ``_weight_offload_internal_budget_bytes`` -> default of
  // ``floor + pinned``) and hard-fails if the result is below
  // ``floor + pinned``.
  static ::executorch::runtime::Result<std::unique_ptr<Session>> create(
      const Payload& payload,
      ::executorch::backends::aoti::AOTIDelegateHandle* handle,
      AOTICatalog catalog,
      const uint8_t* weights_blob,
      cudaStream_t compute_stream,
      ::executorch::runtime::BackendInitContext& context);

  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Hot path. Called from the probe c-shim via the registry
  // callback. ``input`` is the AOTI-provided constant tensor handle
  // (only used here for diagnostics; identity of the FQN comes from
  // ``schedule[probe_id]``).
  ::executorch::runtime::Error serve(
      ::executorch::backends::aoti::Tensor* input,
      int64_t probe_id,
      ::executorch::backends::aoti::Tensor** output);

  // Callback shim that ``ProbeRegistry`` invokes. Casts the
  // ``void* context`` back to ``Session*`` and forwards.
  static ::executorch::runtime::Error registry_callback(
      void* context,
      ::executorch::backends::aoti::Tensor* input,
      int64_t probe_id,
      ::executorch::backends::aoti::Tensor** output);

  // Read-only accessors for tests + the destructor stats line.
  const SessionStats& stats() const {
    return stats_;
  }
  uint64_t peak_live_bytes() const {
    return peak_live_bytes_;
  }
  uint64_t total_budget_bytes() const {
    return total_budget_bytes_;
  }
  uint64_t pinned_bytes_total() const {
    return pinned_bytes_total_;
  }
  uint64_t streaming_budget_bytes() const {
    return streaming_budget_bytes_;
  }
  uint64_t floor_bytes() const {
    return floor_bytes_;
  }

  // Per-FQN host mirror. ``host_ptr`` was allocated via
  // ``cudaHostAlloc`` and is freed in the destructor. Public so the
  // free-on-error helper in ``session.cpp``'s anonymous namespace
  // can reference it; nothing outside that TU should touch it.
  struct HostEntry {
    void* host_ptr{nullptr};
    uint64_t nbytes{0};
    int32_t dtype{0};
    int32_t device_index{0};
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
  };

  // Per-live-allocation tracking. ``ready_event`` is recorded on the
  // copy stream after the H2D completes; consumers must
  // ``cudaStreamWaitEvent`` it before reading. ``lru_it`` points into
  // ``Session::lru_order_`` so a hit splices to the back in O(1)
  // and pick_lru() returns ``live_.find(lru_order_.front())``. Public
  // for the same reason as HostEntry: TU-internal cleanup helpers
  // reference it.
  struct LiveAllocation {
    void* dev_ptr{nullptr};
    uint64_t nbytes{0};
    cudaEvent_t ready_event{nullptr};
    std::list<std::string>::iterator lru_it{};
  };

 private:
  Session() = default;

  // Returns ``live_.end()`` if ``lru_order_`` is empty. Otherwise
  // returns the live_ entry for the front (least-recently-used) FQN
  // — O(1) via lru_order_ + the hash lookup. Pinned entries never
  // enter live_/lru_order_, so they're naturally excluded.
  std::unordered_map<std::string, LiveAllocation>::iterator pick_lru();

  // Best-effort depth-1 prefetch. Called at the end of every
  // successful serve() with the current probe_id. Looks up
  // schedule_[(probe_id + 1) % size()] and, if that FQN is not
  // already live AND not the just-served FQN, attempts to evict +
  // alloc + H2D + record-ready-event for it. Errors are LOGGED and
  // counted in stats_.prefetch_attempted but never returned —
  // the caller (serve) has already populated *output with the
  // current probe's result, and the prefetch is strictly speculative.
  // The next serve() that consumes this FQN sees it as a pool hit
  // and waits on its ready_event before the kernel reads it.
  ::executorch::runtime::Error opportunistic_prefetch(int64_t probe_id);

  // Build a borrowed SlimTensor wrapping the given device pointer
  // with host_entries_[fqn]'s metadata. Used by the hit path, the
  // miss path's post-allocation step, the pinned fast path, and
  // (via prefetch's lookup) the next-hit path. Extracted to one
  // helper so the four sites share the same arg list and error
  // handling.
  //
  // OWNERSHIP: aoti_torch_create_tensor_from_blob_v2 ``new``s a
  // SlimTensor; the AOTI c-shim contract (see
  // backends/cuda/runtime/shims/memory.h and the int4mm output
  // doc comment) makes the CALLER responsible for freeing the
  // returned handle via aoti_torch_delete_tensor_object. For probe
  // outputs that caller is AOTI's generated wrapper.cpp, which
  // wraps the AtenTensorHandle in a RAIIAtenTensorHandle (PyTorch's
  // cpp_wrapper_cpu convention) so the handle is freed when it
  // falls out of scope after the kernel consumes it. ``from_blob``
  // is NON-OWNING — the delete frees only the wrapper, never our
  // pool / pinned / dummy GPU buffer, so the per-serve() ``new
  // SlimTensor`` does not leak the underlying allocation.
  ::executorch::runtime::Error wrap_borrowed_tensor(
      void* dev_ptr,
      const HostEntry& host,
      ::executorch::backends::aoti::Tensor** output);

  // Shared eviction + alloc + H2D path for serve()'s miss and
  // opportunistic_prefetch(). On success: ``*dev_out`` holds a
  // freshly cudaMallocFromPoolAsync'd pool allocation of
  // ``host.nbytes`` bytes, and ``*ready_out`` is a cudaEvent_t
  // recorded on ``copy_stream_`` after the H2D queues. Caller is
  // responsible for: incrementing ``bytes_in_flight_`` +
  // ``peak_live_bytes_``, emplacing into ``live_`` (with the event
  // as the entry's ready_event), bumping its own success counter,
  // and (for serve only) cudaStreamWaitEvent(compute_, ready).
  //
  // Internally: evict LRU victims until ``need`` fits in the
  // streaming budget (skipping ``guard_fqn`` if non-null), update
  // bytes_in_flight_ / live_ / stats_.evictions for each, then
  // event-record + cross-stream wait on the eviction batch, then
  // alloc + memcpy + ready event. On any failure: cudaFreeAsync any
  // ``*dev_out`` already allocated, leave session state consistent,
  // return Error::Internal. ``log_tag`` is the prefix used in every
  // diagnostic the helper emits (e.g. "[ET_WEIGHT_OFFLOAD][ERROR]"
  // or "[ET_WEIGHT_OFFLOAD][WARN] prefetch").
  ::executorch::runtime::Error make_room_and_alloc(
      const std::string& fqn,
      const HostEntry& host,
      const std::string* guard_fqn,
      const char* log_tag,
      void** dev_out,
      cudaEvent_t* ready_out);

  // schedule_[probe_id] is the FQN string; host_entries_[fqn] holds
  // the mirrored bytes + metadata. The two-step indirection is
  // intentional: schedule may repeat the same FQN at distinct probe
  // sites (per-consumer probes), but the host mirror is one entry
  // per unique FQN.
  std::vector<std::string> schedule_;
  std::unordered_map<std::string, HostEntry> host_entries_;
  std::unordered_map<std::string, LiveAllocation> live_;
  cudaStream_t compute_stream_{nullptr};
  cudaStream_t copy_stream_{nullptr};
  cudaMemPool_t pool_{nullptr};
  int32_t device_index_{0};

  // Total budget = user's GPU memory cap = pinned + streaming pool.
  uint64_t total_budget_bytes_{0};
  // Sum of pinned-FQN logical bytes. Subtracted from total to
  // derive the streaming-pool cap.
  uint64_t pinned_bytes_total_{0};
  // Streaming pool cap = total - pinned. The miss/prefetch eviction
  // loops compare ``bytes_in_flight_ + need`` against this, NOT
  // total_budget_bytes_ — pinned allocations live outside the pool
  // (separate cudaMalloc) and don't count toward in-flight.
  uint64_t streaming_budget_bytes_{0};
  uint64_t floor_bytes_{0};
  uint64_t bytes_in_flight_{0};
  uint64_t peak_live_bytes_{0};

  // Intrusive LRU order. front() = least-recently-used FQN
  // (eviction candidate); back() = most-recently-used. Every entry
  // in ``live_`` owns one node here, addressed by
  // ``LiveAllocation::lru_it``. On a hit / fresh miss we splice the
  // entry's node to the back in O(1); on eviction we erase the
  // front node + its live_ entry.
  std::list<std::string> lru_order_;

  std::string method_name_;
  SessionStats stats_;

  // Bookkeeping for tear-down: every pointer we registered with
  // ProbeRegistry so we can unregister precisely on destruction.
  // Stored as ``void*`` to avoid the lookup needing to round-trip
  // through ``host_entries_``.
  std::vector<const void*> registered_dummies_;

  // Owns the 1-byte GPU dummies + SlimTensor wrappers AOTI dispatches
  // probes with. Session::~Session calls
  // ``dummies_.clear_and_free()`` explicitly between "destroy pool"
  // and "free host mirror"; the field's own destructor (which runs
  // implicitly after the explicit cleanup) is a no-op.
  DummyInstallation dummies_;

  // Pinning state. ``pin_set_`` is the O(1) hot-path check
  // Session::serve uses to choose the resident fast path; ``pinned_``
  // owns the GPU buffers, allocated once via ``cudaMalloc`` at
  // Session::create and freed in the dtor between dummies cleanup
  // and host-mirror free.
  std::unordered_set<std::string> pin_set_;
  std::unordered_map<std::string, void*> pinned_;
};

} // namespace executorch::backends::cuda::weight_offload

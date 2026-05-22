/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// ===========================================================================
// EXPERIMENTAL -- PROCESS-GLOBAL PROBE DISPATCH TABLE
// ===========================================================================
// Maps the device pointer AOTI passes into ``aoti_torch_cuda_probe``
// to the offload runtime that should handle it. The c-shim
// (``probe_op.cpp``, linked into ``aoti_cuda_shims``) is the sole
// reader; ``CudaBackend`` ``weight_offload::Session`` is the sole
// writer, registering its dummy pointers at construction and
// unregistering at destruction.
//
// Lookup hits return ``(callback, context)`` and the c-shim
// forwards the probe call. Lookup misses fall back to the identity
// passthrough ONLY when the registry has zero active contexts
// (preserves manual-probe tests that never construct a Session).
// Once any Session is registered, a miss is a hard fail — otherwise
// the runtime could silently read the eager AOTI constant and mask
// a broken pointer binding.
//
// The registry is deliberately tiny: pointer -> (callback,
// context). FQN resolution stays in
// ``Session``'s ``schedule[probe_id]`` lookup; nothing in this
// table knows or cares about FQNs.
//
// Thread safety: ``lookup`` and ``has_any_context`` are read-only and
// take a shared lock, so concurrent probe dispatch across Sessions
// (e.g. prefill and decode methods running on separate threads) does
// not serialise. ``register_entries`` and ``unregister_context`` take
// an exclusive lock and are expected once per Session lifetime.
// ===========================================================================

#include <cstddef>
#include <cstdint>

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/export.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::cuda::weight_offload {

using ServeCallback = ::executorch::runtime::Error (*)(
    void* context,
    ::executorch::backends::aoti::Tensor* input,
    int64_t probe_id,
    ::executorch::backends::aoti::Tensor** output);

struct LookupResult {
  bool found{false};
  ServeCallback callback{nullptr};
  void* context{nullptr};
};

// Class-level ``AOTI_SHIM_EXPORT`` so all methods are visible
// across the ``aoti_cuda_shims`` shared-library boundary on
// Windows/MSVC (the library disables export-all-symbols). On
// Linux/macOS the macro is a no-op and symbol visibility is
// already default. ``session.cpp`` (in ``aoti_cuda_backend``)
// calls these methods through the boundary.
class AOTI_SHIM_EXPORT ProbeRegistry {
 public:
  // Process-global singleton. The state lives in
  // ``aoti_cuda_shims`` so the c-shim accesses it directly without
  // having to dlsym a function pointer.
  static ProbeRegistry& instance();

  // Bulk-register ``dummy_ptrs`` for ``context``. Every pointer in
  // ``dummy_ptrs`` must be unique across all live contexts in the
  // registry; collisions return ``Error::InvalidArgument`` and
  // leave the registry untouched.
  ::executorch::runtime::Error register_entries(
      void* context,
      ServeCallback callback,
      const void* const* dummy_ptrs,
      size_t num_dummy_ptrs);

  // Remove the entries the caller registered. ``dummy_ptrs`` should
  // be the exact pointer set this ``context`` previously passed to
  // ``register_entries``; entries not registered to this context are
  // skipped silently (so a partial register/unregister sequence is
  // safe). O(K) in ``num_dummy_ptrs``, NOT O(N) over the whole
  // registry — important when several Sessions are live and one
  // teardown shouldn't iterate the others' entries. Safe to call
  // even if ``context`` never registered.
  void unregister_context(
      void* context,
      const void* const* dummy_ptrs,
      size_t num_dummy_ptrs);

  // Hot-path lookup. ``found=false`` means the pointer is not in
  // the table; the c-shim's "registry empty?" check distinguishes
  // "no offload at all" (identity fallback) from "offload active
  // but pointer unbound" (hard fail).
  LookupResult lookup(const void* dummy_ptr) const;

  // True iff at least one context is registered.
  bool has_any_context() const;

 private:
  ProbeRegistry();
  ~ProbeRegistry();
  ProbeRegistry(const ProbeRegistry&) = delete;
  ProbeRegistry& operator=(const ProbeRegistry&) = delete;

  struct Impl;
  Impl* impl_;
};

} // namespace executorch::backends::cuda::weight_offload

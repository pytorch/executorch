/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/weight_offload/probe_registry.h>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace executorch::backends::cuda::weight_offload {

// shared_mutex so the hot-path readers (lookup + has_any_context, both
// called per-probe from probe_op.cpp) can run concurrently across
// Sessions. register_entries / unregister_context fire once per
// Session lifetime and take the exclusive lock.
struct ProbeRegistry::Impl {
  mutable std::shared_mutex mu;
  std::unordered_map<const void*, LookupResult> by_ptr;
  std::unordered_set<void*> active_contexts;
};

ProbeRegistry& ProbeRegistry::instance() {
  static ProbeRegistry* singleton = new ProbeRegistry();
  return *singleton;
}

ProbeRegistry::ProbeRegistry() : impl_(new Impl()) {}

ProbeRegistry::~ProbeRegistry() {
  delete impl_;
}

::executorch::runtime::Error ProbeRegistry::register_entries(
    void* context,
    ServeCallback callback,
    const void* const* dummy_ptrs,
    size_t num_dummy_ptrs) {
  using ::executorch::runtime::Error;
  if (context == nullptr || callback == nullptr ||
      (num_dummy_ptrs > 0 && dummy_ptrs == nullptr)) {
    return Error::InvalidArgument;
  }
  std::unique_lock<std::shared_mutex> guard(impl_->mu);
  // First pass: detect collisions WITHOUT mutating, so a failure
  // leaves the registry untouched. ``batch_seen`` makes the
  // intra-batch duplicate check O(N) instead of O(N^2) for the
  // thousand-constant case.
  std::unordered_set<const void*> batch_seen;
  batch_seen.reserve(num_dummy_ptrs);
  for (size_t i = 0; i < num_dummy_ptrs; ++i) {
    const void* p = dummy_ptrs[i];
    if (p == nullptr) {
      return Error::InvalidArgument;
    }
    if (impl_->by_ptr.find(p) != impl_->by_ptr.end()) {
      return Error::InvalidArgument;
    }
    if (!batch_seen.insert(p).second) {
      return Error::InvalidArgument;
    }
  }
  // Second pass: insert.
  for (size_t i = 0; i < num_dummy_ptrs; ++i) {
    impl_->by_ptr.emplace(dummy_ptrs[i], LookupResult{true, callback, context});
  }
  impl_->active_contexts.insert(context);
  return Error::Ok;
}

void ProbeRegistry::unregister_context(
    void* context,
    const void* const* dummy_ptrs,
    size_t num_dummy_ptrs) {
  std::unique_lock<std::shared_mutex> guard(impl_->mu);
  for (size_t i = 0; i < num_dummy_ptrs; ++i) {
    const void* p = dummy_ptrs[i];
    if (p == nullptr) {
      continue;
    }
    auto it = impl_->by_ptr.find(p);
    if (it != impl_->by_ptr.end() && it->second.context == context) {
      impl_->by_ptr.erase(it);
    }
  }
  impl_->active_contexts.erase(context);
}

LookupResult ProbeRegistry::lookup(const void* dummy_ptr) const {
  std::shared_lock<std::shared_mutex> guard(impl_->mu);
  auto it = impl_->by_ptr.find(dummy_ptr);
  if (it == impl_->by_ptr.end()) {
    return LookupResult{};
  }
  return it->second;
}

bool ProbeRegistry::has_any_context() const {
  std::shared_lock<std::shared_mutex> guard(impl_->mu);
  return !impl_->active_contexts.empty();
}

} // namespace executorch::backends::cuda::weight_offload

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/native/NativeBackendInternal.h>

namespace executorch {
namespace backends {
namespace native {

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

} // namespace native
} // namespace backends
} // namespace executorch

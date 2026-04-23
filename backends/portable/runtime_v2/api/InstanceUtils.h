/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/BindingTable.h>
#include <executorch/backends/portable/runtime_v2/api/Buffer.h>
#include <executorch/backends/portable/runtime_v2/api/Event.h>
#include <executorch/backends/portable/runtime_v2/api/GraphTypes.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <string_view>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Helpers shared by all backend Instance implementations.
 *
 * These deliberately operate against the public Event interface (no
 * dynamic_cast to backend-specific subclasses).
 */

// If any event in `wait_for` is in a terminal failure state
// (Failed / Poisoned), poison `signal` with that upstream error and
// return InternalError. Otherwise return Ok.
//
// Rationale: callers should short-circuit their work when an upstream
// dependency failed; the poisoned status propagates the failure
// downstream so consumers don't read garbage.
inline ::executorch::runtime::Error check_async_dependencies(
    ::executorch::runtime::Span<Event* const> wait_for,
    Event* signal) {
  for (Event* dep : wait_for) {
    if (!dep) continue;
    EventStatus s = dep->status();
    if (s == EventStatus::Failed || s == EventStatus::Poisoned) {
      if (signal) signal->signal_poisoned(dep->error());
      return ::executorch::runtime::Error::Internal;
    }
  }
  return ::executorch::runtime::Error::Ok;
}

/**
 * Verify (don't sync) the data_ptr invariant for every tensor arg of every
 * op in a CompiledSegment: TensorImpl::data_ptr MUST equal
 * bindings.get(remap(vid))->host_ptr() for every tensor argument.
 *
 * The data_ptr is established earlier (prebind_owned_buffers /
 * bind_inputs / upload_from_host); by the time we get here it must
 * already be in sync. Any mismatch indicates an upstream bug; we log
 * and signal failure rather than silently overwriting.
 *
 * Templated over CompiledSegment subclass (which exposes
 * `instruction_indices()` and `remap(vid)`); avoids virtual dispatch and
 * a base class.
 *
 * On any violation: logs an Error with `backend_name` prefix, signals
 * `signal->signal_failed(err)` (if non-null), and returns the error.
 *
 * Non-tensor args (scalars, IntLists) are silently skipped — they
 * legitimately don't have Buffer bindings and the kernel reads them
 * from the EValue directly.
 */
template <typename SegmentT>
::executorch::runtime::Error verify_segment_bindings(
    const SegmentT* seg,
    const ::executorch::backends::portable::Graph* graph,
    ::executorch::runtime::Span<::executorch::runtime::EValue> values,
    BindingView bindings,
    Event* signal,
    std::string_view backend_name) {
  using Err = ::executorch::runtime::Error;
  auto fail = [&](Err e) -> Err {
    if (signal) signal->signal_failed(e);
    return e;
  };

  for (uint32_t instr_idx : seg->instruction_indices()) {
    if (instr_idx >= graph->num_instructions()) {
      ET_LOG(Error,
             "%.*s::execute: instruction index %u out of range "
             "(graph has %zu instructions)",
             static_cast<int>(backend_name.size()), backend_name.data(),
             instr_idx, graph->num_instructions());
      return fail(Err::InvalidProgram);
    }
    auto op = graph->get_instruction(instr_idx);

    auto check_arg = [&](uint32_t vid_orig) -> Err {
      uint32_t vid = seg->remap(vid_orig);
      if (vid >= values.size()) {
        ET_LOG(Error,
               "%.*s::execute: value_id=%u out of range "
               "(values.size()=%zu) — router/remap bug",
               static_cast<int>(backend_name.size()), backend_name.data(),
               vid, values.size());
        return fail(Err::InvalidProgram);
      }
      const auto& ev = values[vid];
      // Non-tensor args legitimately have no Buffer binding.
      if (!ev.isTensor()) return Err::Ok;
      Buffer* buf = bindings.get(vid);
      if (!buf) {
        ET_LOG(Error,
               "%.*s::execute: tensor value_id=%u has no Buffer binding "
               "— init bug",
               static_cast<int>(backend_name.size()), backend_name.data(),
               vid);
        return fail(Err::InvalidState);
      }
      void* hp = buf->host_ptr();
      if (!hp) {
        ET_LOG(Error,
               "%.*s::execute: Buffer for value_id=%u has null host_ptr "
               "— Buffer construction bug",
               static_cast<int>(backend_name.size()), backend_name.data(),
               vid);
        return fail(Err::Internal);
      }
      void* current = ev.toTensor().mutable_data_ptr();
      if (current != hp) {
        ET_LOG(Error,
               "%.*s::execute: tensor value_id=%u data_ptr=%p but bound "
               "Buffer host_ptr=%p — sync invariant violated (prebind / "
               "bind_inputs / upload_from_host should have kept these in "
               "sync)",
               static_cast<int>(backend_name.size()), backend_name.data(),
               vid, current, hp);
        return fail(Err::Internal);
      }
      return Err::Ok;
    };

    for (size_t i = 0; i < op.num_inputs(); ++i) {
      if (auto e = check_arg(op.input(i)); e != Err::Ok) return e;
    }
    for (size_t i = 0; i < op.num_outputs(); ++i) {
      if (auto e = check_arg(op.output(i)); e != Err::Ok) return e;
    }
  }
  return Err::Ok;
}

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

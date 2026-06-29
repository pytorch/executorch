/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "mlx_mutable_state.h"

#include "MLXExecutor.h"
#include "MLXLoader.h"

#include <executorch/runtime/platform/log.h>

#include <mutex>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace mlx {

using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

namespace {

struct HandleInfo {
  const MLXProgram* program{nullptr};
  MutableBufferData* default_buffers{nullptr};
};

struct Context {
  // Delegate handles associated with this loaded program (one per loaded
  // method). Keyed by opaque MLXHandle pointer.
  std::unordered_map<const void*, HandleInfo> handles;
  // Per-session mutable buffers: token -> (handle -> buffers). Allocated lazily
  // on first execute for a given (session, handle).
  std::unordered_map<int, std::unordered_map<const void*, MutableBufferData>>
      sessions;
  int next_token{0};
  // Sticky setup failure. Once set (e.g. by nested load scopes), available(),
  // validate_coverage(), create_session(), and rebind fail consistently.
  Error build_error{Error::Ok};
};

// Process-global registry. MLX serializes execution via its own global mutex
// and the engine serializes per session, but the registry itself is guarded
// here so context/session lifecycle calls from other threads are safe.
std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

std::unordered_map<MutableStateContext, Context>& contexts() {
  static std::unordered_map<MutableStateContext, Context> c;
  return c;
}

std::unordered_map<const void*, MutableStateContext>& handle_ctx() {
  static std::unordered_map<const void*, MutableStateContext> m;
  return m;
}

MutableStateContext g_next_ctx = 1; // 0 is reserved as invalid.

// Thread-local load scope and active (ctx, session) selection.
thread_local MutableStateContext tl_loading_ctx = kInvalidMutableContext;
thread_local MutableStateContext tl_active_ctx = kInvalidMutableContext;
thread_local int tl_active_token = kNoMutableSession;

} // namespace

namespace detail {

MutableStateContext mutable_state_create_context() {
  std::lock_guard<std::mutex> g(registry_mutex());
  MutableStateContext ctx = g_next_ctx++;
  if (ctx == kInvalidMutableContext) {
    ctx = g_next_ctx++;
  }
  contexts()[ctx];
  return ctx;
}

void mutable_state_destroy_context(MutableStateContext ctx) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  if (it == contexts().end()) {
    return;
  }
  for (const auto& kv : it->second.handles) {
    handle_ctx().erase(kv.first);
  }
  contexts().erase(it);
}

void mutable_state_begin_load(MutableStateContext ctx) {
  if (tl_loading_ctx != kInvalidMutableContext) {
    // Nested load scopes would silently overwrite the thread-local association.
    // Mark both the already-active and the new context invalid instead.
    std::lock_guard<std::mutex> g(registry_mutex());
    auto active = contexts().find(tl_loading_ctx);
    if (active != contexts().end()) {
      active->second.build_error = Error::InvalidState;
    }
    auto nested = contexts().find(ctx);
    if (nested != contexts().end()) {
      nested->second.build_error = Error::InvalidState;
    }
    ET_LOG(Error, "mutable_state: nested load scopes are not supported");
    tl_loading_ctx = kInvalidMutableContext;
    return;
  }
  tl_loading_ctx = ctx;
}

void mutable_state_end_load() {
  tl_loading_ctx = kInvalidMutableContext;
}

bool mutable_state_available(MutableStateContext ctx) {
  if (ctx == kInvalidMutableContext) {
    return false;
  }
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  return it != contexts().end() && it->second.build_error == Error::Ok &&
      !it->second.handles.empty();
}

int64_t mutable_state_bytes_per_session(MutableStateContext ctx) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  if (it == contexts().end()) {
    return 0;
  }
  int64_t total = 0;
  // Per-session mutable buffers are allocated from program metadata
  // (mutable_buffer_map + tensor_meta), independent of whether the handle kept
  // a default copy. Compute the estimate from metadata so it stays correct even
  // when default mutable-buffer init was skipped (skip_mutable_buffer_init).
  for (const auto& kv : it->second.handles) {
    const MLXProgram* program = kv.second.program;
    if (program == nullptr) {
      continue;
    }
    for (const auto& slot : program->mutable_buffer_map) {
      if (slot.slot_type != SlotType::TensorSlot ||
          slot.idx >= program->tensor_meta.size() ||
          !program->tensor_meta[slot.idx].has_value()) {
        continue;
      }
      const auto& meta = *program->tensor_meta[slot.idx];
      // Sum sizes from metadata, clamping each tensor to kMaxAllocationBytes so
      // malformed (oversized) shapes in an untrusted program can't overflow the
      // accumulator. Real allocations are independently bounded by
      // check_allocation_bounded at load_mutable_buffers.
      uint64_t bytes = static_cast<uint64_t>(
          ::mlx::core::size_of(resolve_dtype(meta.scalar_type)));
      bool dynamic = false;
      for (const auto& dim : meta.shape) {
        if (dim.value < 0) {
          dynamic = true;
          break;
        }
        const uint64_t d = static_cast<uint64_t>(dim.value);
        if (d == 0) {
          bytes = 0;
          break;
        }
        if (bytes > kMaxAllocationBytes / d) {
          bytes = kMaxAllocationBytes; // clamp; avoids overflow
        } else {
          bytes *= d;
        }
      }
      if (dynamic) {
        continue;
      }
      if (bytes > kMaxAllocationBytes) {
        bytes = kMaxAllocationBytes;
      }
      total += static_cast<int64_t>(bytes);
    }
  }
  return total;
}

Error mutable_state_validate_coverage(MutableStateContext ctx) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  if (it == contexts().end()) {
    return Error::InvalidArgument;
  }
  if (it->second.build_error != Error::Ok) {
    return it->second.build_error;
  }
  // MLX clones all mutable buffers by tid; there is no FQN coverage to verify.
  return Error::Ok;
}

Result<int> mutable_state_create_session(MutableStateContext ctx) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  if (it == contexts().end()) {
    ET_LOG(Error, "mutable_state_create_session: unknown context %d", ctx);
    return Error::InvalidState;
  }
  Context& c = it->second;
  if (c.build_error != Error::Ok) {
    return c.build_error;
  }
  if (c.handles.empty()) {
    ET_LOG(
        Error, "mutable_state_create_session: no backend handles registered");
    return Error::NotSupported;
  }
  int token = c.next_token++;
  // Per-handle buffers are allocated lazily on first execute.
  c.sessions[token];
  return token;
}

void mutable_state_destroy_session(MutableStateContext ctx, int token) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(ctx);
  if (it == contexts().end()) {
    return;
  }
  it->second.sessions.erase(token);
}

void mutable_state_set_active(MutableStateContext ctx, int token) {
  tl_active_ctx = ctx;
  tl_active_token = token;
}

} // namespace detail

void mutable_state_note_handle(
    const void* handle,
    const MLXProgram* program,
    MutableBufferData* default_buffers) {
  if (tl_loading_ctx == kInvalidMutableContext) {
    return; // No multi-session owner active during this load: single-session.
  }
  std::lock_guard<std::mutex> g(registry_mutex());
  auto it = contexts().find(tl_loading_ctx);
  if (it == contexts().end()) {
    return;
  }
  it->second.handles[handle] = HandleInfo{program, default_buffers};
  handle_ctx()[handle] = tl_loading_ctx;
}

void mutable_state_forget_handle(const void* handle) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto hit = handle_ctx().find(handle);
  if (hit == handle_ctx().end()) {
    return;
  }
  auto cit = contexts().find(hit->second);
  if (cit != contexts().end()) {
    cit->second.handles.erase(handle);
    for (auto& session : cit->second.sessions) {
      session.second.erase(handle);
    }
  }
  handle_ctx().erase(hit);
}

Error mutable_state_rebind_for_execute(
    const void* handle,
    ExecutionState& state) {
  std::lock_guard<std::mutex> g(registry_mutex());
  auto hit = handle_ctx().find(handle);
  if (hit == handle_ctx().end()) {
    if (tl_active_token != kNoMutableSession) {
      ET_LOG(
          Error,
          "mutable_state_rebind_for_execute: active session set but handle has "
          "no mutable-state context");
      return Error::Internal;
    }
    // Handle was not loaded under a multi-session owner: keep default buffers.
    return Error::Ok;
  }
  auto cit = contexts().find(hit->second);
  if (cit == contexts().end()) {
    return Error::Ok;
  }
  Context& ctx = cit->second;
  if (ctx.build_error != Error::Ok) {
    return ctx.build_error;
  }
  // Invariant: a handle present in handle_ctx() is present in ctx.handles. Look
  // it up explicitly (not operator[]) so a broken invariant fails loudly
  // instead of inserting a {nullptr, nullptr} entry that later null-derefs in
  // load_mutable_buffers(*info.program, ...).
  auto info_it = ctx.handles.find(handle);
  if (info_it == ctx.handles.end()) {
    ET_LOG(
        Error,
        "mutable_state_rebind_for_execute: handle has a context but no "
        "registered HandleInfo (invariant broken)");
    return Error::Internal;
  }
  HandleInfo& info = info_it->second;

  const bool has_active_session = tl_active_token != kNoMutableSession;
  const bool active_for_this_ctx =
      has_active_session && tl_active_ctx == hit->second;

  // A session is active, but for a different context than the one this handle
  // belongs to. Falling back to default buffers would silently execute with the
  // wrong model/session state, so refuse instead.
  if (has_active_session && !active_for_this_ctx) {
    ET_LOG(
        Error,
        "mutable_state_rebind_for_execute: active context mismatch (a session "
        "is active for a different loaded program than the one executing)");
    return Error::Internal;
  }

  if (!active_for_this_ctx) {
    // No session selected. Refuse if sessions exist (running against the
    // default buffers here would not isolate state from created sessions).
    if (!ctx.sessions.empty()) {
      ET_LOG(
          Error,
          "mutable_state_rebind_for_execute: no active session selected but "
          "sessions exist for this program");
      return Error::InvalidState;
    }
    state.mutable_buffers = info.default_buffers;
    return Error::Ok;
  }

  auto sit = ctx.sessions.find(tl_active_token);
  if (sit == ctx.sessions.end()) {
    ET_LOG(
        Error,
        "mutable_state_rebind_for_execute: unknown session token %d",
        tl_active_token);
    return Error::InvalidState;
  }

  auto& per_handle = sit->second;
  auto bit = per_handle.find(handle);
  if (bit == per_handle.end()) {
    // First execute for this (session, handle): allocate fresh zeroed buffers.
    // Constants/weights stay shared (ExecutionState::constants is untouched);
    // only the mutable buffers are per-session.
    MutableBufferData buffers;
    try {
      load_mutable_buffers(*info.program, buffers);
    } catch (const std::exception& e) {
      ET_LOG(
          Error,
          "mutable_state_rebind_for_execute: failed to allocate session "
          "buffers: %s",
          e.what());
      return Error::MemoryAllocationFailed;
    }
    bit = per_handle.emplace(handle, std::move(buffers)).first;
  }
  // unordered_map keeps element pointers stable across rehash, so this remains
  // valid for the duration of the execute.
  state.mutable_buffers = &bit->second;
  return Error::Ok;
}

} // namespace mlx
} // namespace backends
} // namespace executorch

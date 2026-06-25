/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <utility>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

// MLX-private support for running one loaded MLX program with multiple isolated
// instances of its mutable buffers (KV cache, conv/recurrent state). Callers
// create sessions and execute with one active session selected.
//
// Unlike the CUDA backend, the MLX runtime owns mutable buffers directly in a
// swappable container (ExecutionState::mutable_buffers is a
// MutableBufferData*), so per-session isolation is a pointer swap to a fresh
// MutableBufferData — no FQN registration / constant-repoint hook is needed.

namespace executorch {
namespace backends {
namespace mlx {

// Forward declarations (defined in MLXLoader.h / MLXExecutor.h).
struct MLXProgram;
struct MutableBufferData;
struct ExecutionState;

// Opaque per-loaded-program context id (0 = invalid).
using MutableStateContext = int;
constexpr MutableStateContext kInvalidMutableContext = 0;

// Sentinel for execution without per-session rebinding.
constexpr int kNoMutableSession = -1;

// Implementation entry points. Callers should use MutableStateContextOwner.
namespace detail {

MutableStateContext mutable_state_create_context();
void mutable_state_destroy_context(MutableStateContext ctx);
void mutable_state_begin_load(MutableStateContext ctx);
void mutable_state_end_load();
bool mutable_state_available(MutableStateContext ctx);
int64_t mutable_state_bytes_per_session(MutableStateContext ctx);
::executorch::runtime::Error mutable_state_validate_coverage(
    MutableStateContext ctx);
::executorch::runtime::Result<int> mutable_state_create_session(
    MutableStateContext ctx);
void mutable_state_destroy_session(MutableStateContext ctx, int token);
void mutable_state_set_active(MutableStateContext ctx, int token);

} // namespace detail

// Caller-facing owner for one mutable-state context. Mirrors the CUDA backend's
// MutableStateContextOwner so the example engine can use a symmetric API.
class ET_EXPERIMENTAL MutableStateContextOwner final {
  class LoadScope final {
   public:
    explicit LoadScope(MutableStateContext ctx) {
      detail::mutable_state_begin_load(ctx);
    }

    ~LoadScope() {
      detail::mutable_state_end_load();
    }

    LoadScope(const LoadScope&) = delete;
    LoadScope& operator=(const LoadScope&) = delete;
  };

  class ActiveSessionScope final {
   public:
    ActiveSessionScope(MutableStateContext ctx, int token) {
      detail::mutable_state_set_active(ctx, token);
    }

    ~ActiveSessionScope() {
      detail::mutable_state_set_active(
          kInvalidMutableContext, kNoMutableSession);
    }

    ActiveSessionScope(const ActiveSessionScope&) = delete;
    ActiveSessionScope& operator=(const ActiveSessionScope&) = delete;
  };

 public:
  MutableStateContextOwner() : ctx_(detail::mutable_state_create_context()) {}

  ~MutableStateContextOwner() {
    destroy();
  }

  MutableStateContextOwner(const MutableStateContextOwner&) = delete;
  MutableStateContextOwner& operator=(const MutableStateContextOwner&) = delete;

  MutableStateContextOwner(MutableStateContextOwner&& other) noexcept
      : ctx_(std::exchange(other.ctx_, kInvalidMutableContext)) {}

  MutableStateContextOwner& operator=(
      MutableStateContextOwner&& other) noexcept {
    if (this != &other) {
      destroy();
      ctx_ = std::exchange(other.ctx_, kInvalidMutableContext);
    }
    return *this;
  }

  MutableStateContext get() const {
    return ctx_;
  }

  explicit operator bool() const {
    return ctx_ != kInvalidMutableContext;
  }

  // Associates delegate handles created by `fn` with this context.
  template <typename Fn>
  auto with_load_scope(Fn&& fn) const -> decltype(std::forward<Fn>(fn)()) {
    LoadScope scope(ctx_);
    return std::forward<Fn>(fn)();
  }

  // Selects this context/session while `fn` executes. The caller is responsible
  // for serializing execution that touches the same loaded program.
  //
  // Thread-safety contract: destroy_session()/forget_handle() only take the
  // registry mutex, while rebind (under with_active_session) hands execute a
  // raw pointer into Context::sessions that is dereferenced after the lock is
  // released. The caller must therefore guarantee a session is never destroyed
  // while it is the active session mid-execute (the engine upholds this: a
  // session's buffers are freed only when its owning LLMSession drops, never
  // concurrently with its own execute). Destroying *other* sessions
  // concurrently is safe — unordered_map keeps element pointers stable across
  // rehash.
  template <typename Fn>
  auto with_active_session(int token, Fn&& fn) const
      -> decltype(std::forward<Fn>(fn)()) {
    ActiveSessionScope scope(ctx_, token);
    return std::forward<Fn>(fn)();
  }

  // True only after this context has been associated with at least one loaded
  // MLX backend handle can create isolated mutable-buffer sessions.
  bool available() const {
    return detail::mutable_state_available(ctx_);
  }

  int64_t bytes_per_session() const {
    return detail::mutable_state_bytes_per_session(ctx_);
  }

  ::executorch::runtime::Error validate_coverage() const {
    return detail::mutable_state_validate_coverage(ctx_);
  }

  // Creates an isolated mutable-buffer session for this context.
  // Fails if no loaded MLX backend handle has been associated with the context.
  ET_NODISCARD ::executorch::runtime::Result<int> create_session() const {
    return detail::mutable_state_create_session(ctx_);
  }

  void destroy_session(int token) const {
    detail::mutable_state_destroy_session(ctx_, token);
  }

 private:
  void destroy() {
    if (ctx_ != kInvalidMutableContext) {
      detail::mutable_state_destroy_context(ctx_);
      ctx_ = kInvalidMutableContext;
    }
  }

  MutableStateContext ctx_ = kInvalidMutableContext;
};

// --- MLXBackend hooks --------------------------------------------------------
//
// Called from MLXBackend init/execute/destroy. `handle` is an opaque key (the
// MLXHandle pointer). `program` and `default_buffers` are the handle's own
// program and (init-time) mutable buffers; the manager swaps in per-session
// buffers (or restores the default) by re-pointing `state.mutable_buffers`.

void mutable_state_note_handle(
    const void* handle,
    const MLXProgram* program,
    MutableBufferData* default_buffers);

void mutable_state_forget_handle(const void* handle);

::executorch::runtime::Error mutable_state_rebind_for_execute(
    const void* handle,
    ExecutionState& state);

} // namespace mlx
} // namespace backends
} // namespace executorch

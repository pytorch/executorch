/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

// CUDA-private support for running one loaded CUDA program with multiple
// isolated instances of its mutable buffers. Callers register mutable-buffer
// FQNs, create sessions, and execute with one active session selected.

namespace executorch {
namespace backends {
namespace cuda {

struct CudaDelegateHandle;

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
void mutable_state_register_fqns(
    MutableStateContext ctx,
    const std::vector<std::string>& fqns);
bool mutable_state_available(MutableStateContext ctx);
int64_t mutable_state_bytes_per_session(MutableStateContext ctx);
::executorch::runtime::Error mutable_state_validate_coverage(
    MutableStateContext ctx);
::executorch::runtime::Result<int> mutable_state_create_session(
    MutableStateContext ctx);
void mutable_state_destroy_session(MutableStateContext ctx, int token);
void mutable_state_set_active(MutableStateContext ctx, int token);

} // namespace detail

// Caller-facing owner for one mutable-state context.
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

  void register_fqns(const std::vector<std::string>& fqns) const {
    detail::mutable_state_register_fqns(ctx_, fqns);
  }

  // Associates delegate handles created by `fn` with this context. Register
  // FQNs before entering the load scope.
  template <typename Fn>
  auto with_load_scope(Fn&& fn) const -> decltype(std::forward<Fn>(fn)()) {
    LoadScope scope(ctx_);
    return std::forward<Fn>(fn)();
  }

  // Selects this context/session while `fn` executes. The caller is responsible
  // for serializing execution that touches the same loaded program.
  template <typename Fn>
  auto with_active_session(int token, Fn&& fn) const
      -> decltype(std::forward<Fn>(fn)()) {
    ActiveSessionScope scope(ctx_, token);
    return std::forward<Fn>(fn)();
  }

  bool available() const {
    return detail::mutable_state_available(ctx_);
  }

  int64_t bytes_per_session() const {
    return detail::mutable_state_bytes_per_session(ctx_);
  }

  ::executorch::runtime::Error validate_coverage() const {
    return detail::mutable_state_validate_coverage(ctx_);
  }

  ::executorch::runtime::Result<int> create_session() const {
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

// --- CudaBackend hooks -------------------------------------------------------

void mutable_state_note_handle(CudaDelegateHandle* handle);

void mutable_state_forget_handle(CudaDelegateHandle* handle);

::executorch::runtime::Error mutable_state_rebind_for_execute(
    CudaDelegateHandle* handle);

} // namespace cuda
} // namespace backends
} // namespace executorch

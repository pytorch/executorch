/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Event.h>

#include <executorch/runtime/core/error.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace executorch {
namespace backends {
namespace native {

/**
 * CPU-side Event. Backed by an atomic status flag plus a condition
 * variable for efficient waits when an off-thread producer signals it.
 *
 * Two access patterns:
 *
 *   (a) Synchronous: producer is on the same thread as the consumer
 *       (typical CPU pipeline). signal_complete() returns before any
 *       wait_until_settled() runs; the wait sees Complete on the very
 *       first status() check and returns immediately.
 *
 *   (b) Cross-thread: producer is on another thread (e.g., a GPU
 *       completion handler invoking signal_complete from Metal's
 *       background dispatch queue). The waiter does a short bounded
 *       spin (sub-microsecond, captures the producer-finishes-soon
 *       case) then sleeps on the condition variable.
 *
 * Memory ordering: status_ is atomic with acquire/release pairs.
 * Errors set before the status store; readers acquire the status
 * before reading error_.
 */
class CpuEvent final : public Event {
 public:
  CpuEvent() = default;

  void prepare_signal() override {
    // Allowed only when not Failed/Poisoned (executor enforces).
    error_ = ::executorch::runtime::Error::Ok;
    status_.store(EventStatus::Pending, std::memory_order_release);
  }

  EventStatus status() const override {
    return status_.load(std::memory_order_acquire);
  }

  ::executorch::runtime::Error error() const override {
    return error_;
  }

  void signal_complete() override {
    settle(EventStatus::Complete, ::executorch::runtime::Error::Ok);
  }

  void signal_failed(::executorch::runtime::Error e) override {
    settle(EventStatus::Failed, e);
  }

  void signal_poisoned(::executorch::runtime::Error upstream_error) override {
    settle(EventStatus::Poisoned, upstream_error);
  }

  // Bounded spin (sub-microsecond, free if producer finishes during
  // the spin) then condvar wait. Wakeup latency in the slow path is
  // ~1–10 μs (kernel context switch). Used by CpuEngine::wait().
  void wait_until_settled() override {
    // Fast path 1: already settled, no synchronization needed.
    if (status() != EventStatus::Pending) return;

    // Fast path 2: short bounded spin. Captures producers that
    // signal within a few hundred nanoseconds of our call.
    constexpr int kSpinIters = 256;
    for (int i = 0; i < kSpinIters; ++i) {
      if (status() != EventStatus::Pending) return;
#if defined(__x86_64__) || defined(_M_X64)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
      asm volatile("yield");
#endif
    }

    // Slow path: sleep on condvar until a signal_* call wakes us.
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this] {
      return status_.load(std::memory_order_acquire) != EventStatus::Pending;
    });
  }

 private:
  void settle(EventStatus s, ::executorch::runtime::Error e) {
    {
      // Hold the mutex around the store so wait_until_settled()'s
      // predicate can't observe Pending after we've moved past it
      // without seeing the cv notify (lost-wakeup avoidance).
      std::lock_guard<std::mutex> g(mu_);
      error_ = e;
      status_.store(s, std::memory_order_release);
    }
    cv_.notify_all();
  }

  std::atomic<EventStatus> status_{EventStatus::Pending};
  ::executorch::runtime::Error error_ = ::executorch::runtime::Error::Ok;
  // Synchronization for cross-thread wait_until_settled().
  std::mutex mu_;
  std::condition_variable cv_;
};

} // namespace native
} // namespace backends
} // namespace executorch

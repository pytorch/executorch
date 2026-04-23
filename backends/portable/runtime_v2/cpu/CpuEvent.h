/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime_v2/api/Event.h>

#include <executorch/runtime/core/error.h>

#include <atomic>

namespace executorch {
namespace backends {
namespace portable_v2 {

/**
 * Trivially-completed Event for CPU. CPU operations run synchronously,
 * so the producing call sets status to Complete before returning.
 *
 * Memory ordering: status_ is atomic with acquire/release semantics
 * (acquire on read, release on signal) per §4.6.
 */
class CpuEvent final : public Event {
 public:
  CpuEvent() = default;

  void prepare_signal() override {
    // Allowed only when not Failed/Poisoned (executor enforces).
    status_.store(EventStatus::Pending, std::memory_order_release);
    error_ = ::executorch::runtime::Error::Ok;
  }

  EventStatus status() const override {
    return status_.load(std::memory_order_acquire);
  }

  ::executorch::runtime::Error error() const override { return error_; }

  // CPU-specific helpers used by CpuInstance to drive transitions:

  void signal_complete() override {
    status_.store(EventStatus::Complete, std::memory_order_release);
  }

  void signal_failed(::executorch::runtime::Error e) override {
    error_ = e;
    status_.store(EventStatus::Failed, std::memory_order_release);
  }

  void signal_poisoned(::executorch::runtime::Error upstream_error) override {
    error_ = upstream_error;
    status_.store(EventStatus::Poisoned, std::memory_order_release);
  }

 private:
  std::atomic<EventStatus> status_{EventStatus::Pending};
  ::executorch::runtime::Error error_ = ::executorch::runtime::Error::Ok;
};

}  // namespace portable_v2
}  // namespace backends
}  // namespace executorch

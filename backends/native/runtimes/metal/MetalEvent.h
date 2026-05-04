/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/native/core/Event.h>

#include <atomic>

namespace executorch {
namespace backends {
namespace native {

/**
 * MetalEvent — async event signaled by a Metal command buffer's
 * completion handler.
 *
 * Same atomic-flag pattern as CpuEvent. The only difference is who
 * calls signal_complete() / signal_failed():
 *   - CpuEvent: synchronous code right after the work returns.
 *   - MetalEvent: a [commandBuffer addCompletedHandler:] block that
 *     fires on a private dispatch queue after the GPU finishes.
 *
 * The contract is the same: by the time status() observes Complete,
 * both the bytes in any Buffer the producer wrote AND the shape on any
 * TensorImpl the producer's outputs reference are valid for host reads.
 *
 * For MetalEvent specifically, "valid bytes" needs no extra work on
 * Apple Silicon (unified memory: GPU writes are visible to host once
 * the command buffer completes). Shape is updated inside the producer
 * Engine's execute() before signaling, per the standard contract.
 */
class MetalEvent final : public Event {
 public:
  MetalEvent()
      : status_(EventStatus::Complete),
        error_(::executorch::runtime::Error::Ok) {}
  ~MetalEvent() override = default;

  void prepare_signal() override {
    error_ = ::executorch::runtime::Error::Ok;
    status_.store(EventStatus::Pending, std::memory_order_release);
  }

  EventStatus status() const override {
    return status_.load(std::memory_order_acquire);
  }

  ::executorch::runtime::Error error() const override {
    return error_;
  }

  // Called by MetalEngine code (typically inside a command buffer
  // completion handler).
  void signal_complete() override {
    status_.store(EventStatus::Complete, std::memory_order_release);
  }

  void signal_failed(::executorch::runtime::Error err) override {
    error_ = err;
    status_.store(EventStatus::Failed, std::memory_order_release);
  }

  void signal_poisoned(::executorch::runtime::Error err) override {
    error_ = err;
    status_.store(EventStatus::Poisoned, std::memory_order_release);
  }

 private:
  std::atomic<EventStatus> status_;
  ::executorch::runtime::Error error_;
};

} // namespace native
} // namespace backends
} // namespace executorch

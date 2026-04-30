/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>

#include <cstdint>

namespace executorch {
namespace backends {
namespace native {

/**
 * Async coordination primitives. See §4.6 of the design doc.
 */

// Opaque index into Plan::events.
using EventId = uint16_t;
inline constexpr EventId kNoEvent = 0xFFFF;

// Sticky completion status. Once non-Pending, only prepare_signal() can
// clear it.
enum class EventStatus : uint8_t {
  Pending, // not yet signaled
  Complete, // signaled successfully
  Failed, // producer reported a backend error
  Poisoned, // an upstream event in this event's wait_for chain Failed;
            //   this step short-circuited and never executed
};

/**
 * Backend-defined opaque async-completion handle.
 *   CPU: trivially-completed flag.
 *   Metal: MTLEvent (monotonic signal value).
 *   Vulkan: timeline VkSemaphore (host wait via VkFence).
 */
class Event {
 public:
  virtual ~Event() = default;

  // Prepare the event to receive a new signal in this execute() pass. On
  // monotonic backends, advances the expected next-signal value. On
  // reset-style backends, resets the fence. On CPU, clears the flag.
  // Pre-condition: status() != Failed && status() != Poisoned (executor
  // enforces by refusing to start the next execute() if delegate is
  // Poisoned).
  virtual void prepare_signal() = 0;

  // Cheap, lock-free read. Memory ordering: ACQUIRE on the loaded status;
  // pairs with the release-store inside the producing signal path.
  virtual EventStatus status() const = 0;

  bool is_complete() const {
    return status() == EventStatus::Complete;
  }

  // Valid iff status() == Failed or Poisoned.
  virtual ::executorch::runtime::Error error() const = 0;

  // Signaling — promoted to base so call sites can drive any Event*
  // without dynamic_cast'ing to a concrete subclass.
  //
  //   signal_complete: producer finished successfully.
  //   signal_failed:   producer encountered a backend error; propagate
  //                    `e` as the event's error().
  //   signal_poisoned: an upstream event in this event's wait_for chain
  //                    Failed, so this step short-circuited and never
  //                    actually executed; `upstream_error` is the
  //                    upstream's error() value.
  virtual void signal_complete() = 0;
  virtual void signal_failed(::executorch::runtime::Error e) = 0;
  virtual void signal_poisoned(::executorch::runtime::Error upstream_error) = 0;

  // Block the calling thread until status() reaches a terminal state
  // (Complete / Failed / Poisoned). Default impl is a bounded spin
  // followed by a yield-loop — adequate for sync-by-construction
  // backends where the producer settled the event before its caller
  // even saw the event handle. CpuEvent overrides with a condition
  // variable for true async producers (e.g., a Metal completion
  // handler signaling a CpuEvent from a background thread).
  virtual void wait_until_settled() {
    while (status() == EventStatus::Pending) {
#if defined(__x86_64__) || defined(_M_X64)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
      asm volatile("yield");
#endif
    }
  }
};

} // namespace native
} // namespace backends
} // namespace executorch

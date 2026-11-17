/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Host stand-in for esp_cpu_get_cycle_count(): a 240 MHz count of a monotonic
// clock, wrapping like the 32-bit cycle counter.

#pragma once

#include <chrono>
#include <cstdint>

inline uint32_t esp_cpu_get_cycle_count() {
  const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  return static_cast<uint32_t>(ns * 240 / 1000);
}

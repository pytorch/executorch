/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "muse_glimmer_batching_workload.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace muse_glimmer_ci {

struct SweepShape {
  int decodes;
  int chunk;
  int prefill;
  int backlog;

  int tokens() const {
    return decodes + prefill;
  }
};

inline SweepShape sweep_shape(
    int requested,
    int decode_cap,
    int prefill_chunk,
    int max_sessions) {
  if (requested <= 0 || decode_cap <= 0 || prefill_chunk < 0 ||
      max_sessions <= 0) {
    throw std::invalid_argument("invalid sweep width or chunk limits");
  }
  const int decodes =
      requested <= 64 ? requested : std::min(decode_cap, requested - 1);
  const int budget = requested - decodes;
  const int cap = prefill_chunk > 0 ? prefill_chunk : budget;
  const int per_step = budget > 0 ? 1 + (budget - 1) / cap : 0;
  check_session_capacity(
      static_cast<std::size_t>(decodes) + 2ULL * per_step, max_sessions);
  const int chunk = per_step > 0 ? budget / per_step : 0;
  return {decodes, chunk, per_step * chunk, 2 * per_step};
}

struct RequestTiming {
  using Clock = std::chrono::steady_clock;
  Clock::time_point submitted;
  std::optional<double> ttft_ms;

  void first_text(Clock::time_point now) {
    if (!ttft_ms) {
      ttft_ms =
          std::chrono::duration<double, std::milli>(now - submitted).count();
    }
  }
};

inline void check_measurement(
    bool timed_out,
    std::size_t measured,
    std::size_t requested,
    std::size_t failed,
    bool submission_failed) {
  if (timed_out || measured != requested || failed > 0 || submission_failed) {
    throw std::runtime_error(
        "benchmark workload failed or measurement incomplete");
  }
}

} // namespace muse_glimmer_ci

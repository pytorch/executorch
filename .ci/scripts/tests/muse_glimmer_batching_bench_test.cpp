/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "../muse_glimmer_batching_bench_utils.h"

#include <iostream>
#include <limits>

namespace {
void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("CI benchmark regression test failed");
  }
}

template <typename F>
void rejected(F action) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception&) {
    failed = true;
  }
  require(failed);
}
} // namespace

int main() {
  try {
    using namespace muse_glimmer_ci;
    check_session_capacity(4, 4);
    rejected([] { check_session_capacity(5, 4); });
    rejected([] { check_session_capacity(0, 4); });
    rejected([] { check_session_capacity(1, 0); });
    const auto small = sweep_shape(4, 32, 0, 4);
    require(small.decodes == 4 && small.prefill == 0 && small.backlog == 0);
    rejected([] { sweep_shape(8, 32, 0, 4); });
    rejected([] { sweep_shape(128, 32, 0, 33); });
    const auto mixed = sweep_shape(128, 32, 0, 34);
    require(mixed.decodes == 32 && mixed.chunk == 96 && mixed.tokens() == 128);
    require(mixed.backlog == 2);
    const auto chunks = sweep_shape(128, 32, 32, 38);
    require(chunks.chunk == 32 && chunks.backlog == 6);
    rejected([] { sweep_shape(0, 32, 0, 64); });
    rejected([] { sweep_shape(128, 0, 0, 64); });
    rejected([] { sweep_shape(128, 32, -1, 64); });
    rejected([] { sweep_shape(4, 32, 0, -1); });
    rejected([] { sweep_shape(std::numeric_limits<int>::max(), 1, 1, 64); });

    const auto input = [](batching::SessionId session, std::size_t size = 1) {
      return batching::Input{
          session,
          true,
          0,
          size,
          std::make_shared<const std::vector<batching::Token>>(size, 1000),
          0};
    };
    BatchEvidence evidence;
    evidence.observe({{input(1, 100)}});
    evidence.observe({{input(2)}});
    require(evidence.max_sessions == 1 && evidence.multi_session_steps == 0);
    evidence.observe({{input(1), input(1)}});
    require(evidence.multi_session_steps == 0);
    evidence.observe({{input(1), input(2)}});
    require(evidence.max_sessions == 2 && evidence.multi_session_steps == 1);
    require(evidence.mixed_steps == 0);
    evidence.observe({{input(2), input(3, 9)}});
    require(evidence.multi_session_steps == 2 && evidence.mixed_steps == 1);

    const auto submitted = RequestTiming::Clock::time_point{};
    RequestTiming timing{submitted, std::nullopt};
    require(!timing.ttft_ms);
    timing.first_text(submitted + std::chrono::milliseconds(125));
    require(timing.ttft_ms == 125.0);
    timing.first_text(submitted + std::chrono::milliseconds(300));
    require(timing.ttft_ms == 125.0);

    check_measurement(false, 20, 20, 0, false);
    rejected([] { check_measurement(true, 20, 20, 0, false); });
    rejected([] { check_measurement(false, 19, 20, 0, false); });
    rejected([] { check_measurement(false, 20, 20, 1, false); });
    rejected([] { check_measurement(false, 20, 20, 0, true); });
    std::cout << "CI batching workload/metrics tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

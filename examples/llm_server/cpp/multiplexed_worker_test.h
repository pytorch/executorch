/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <functional>

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>

namespace executorch::examples::llm_server::testing {

// Internal, per-worker checkpoints. Callbacks must be bounded and run unlocked.
// BeforeBind/Bound bracket actual generation handle assignment, not
// publication. TerminalEnqueued runs inside the generation or lifecycle
// completion callback. OverflowLatched follows the overflow/cancellation latch
// and queue accounting.
enum class Checkpoint {
  Admitted,
  BeforeBind,
  Bound,
  TerminalEnqueued,
  OverflowLatched,
  TerminalPublished,
};

struct WorkerTestHooks {
  std::function<void(Checkpoint, std::uint64_t)> checkpoint;
  // Observation only, unlocked after actual successful handle assignment.
  std::function<
      void(std::uint64_t, const extension::llm::serving::RequestHandle&)>
      handle_bound;
  // Called unlocked before a final-byte write. Zero uses the real write;
  // otherwise simulate a failed write with this errno to exercise retries.
  std::function<int(std::uint64_t)> terminal_write_error;
};

int run_multiplexed_worker(
    extension::llm::serving::ServingRuntime& runtime,
    int input_fd,
    int output_fd,
    MultiplexedWorkerConfig config,
    WorkerTestHooks hooks);

} // namespace executorch::examples::llm_server::testing

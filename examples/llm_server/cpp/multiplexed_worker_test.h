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

// Internal, per-worker checkpoints. Callbacks must be bounded. TerminalDeferred
// is observation-only under the worker mutex; all other callbacks run unlocked.
enum class Checkpoint {
  Admitted,
  BeforeBind,
  Bound,
  TerminalEnqueued,
  TerminalDeferred,
  TerminalPublished,
};

struct WorkerTestHooks {
  std::function<void(Checkpoint, std::uint64_t)> checkpoint;
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

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstddef>

#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch::examples::llm_server {

struct ET_EXPERIMENTAL MultiplexedWorkerConfig {
  std::size_t max_inflight_requests = 64;
  // Includes the terminating newline. Applies to input and output records.
  std::size_t max_frame_bytes = 1024 * 1024;
  std::size_t token_frames_per_request = 64;
  std::size_t token_bytes_per_request = 256 * 1024;
  std::chrono::milliseconds write_timeout{10000};
  std::chrono::milliseconds startup_timeout{30000};
};

// POSIX JSONL transport. Borrows both descriptors exclusively for the call;
// does not close them. The caller must ignore SIGPIPE. Uses one output writer;
// runtime callbacks enqueue generation and lifecycle results. EOF shuts down
// the runtime (including callback and physical Runner cleanup). Returns 0 on
// clean EOF, 1 on invalid framing, startup failure, or a failed/stalled output
// stream. A usable request_id is required to isolate a malformed request;
// uncorrelatable frames terminate the transport. No callbacks perform
// descriptor I/O. Accepted close/reset ACKs follow earlier same-key generation
// callbacks and capture cleanup, and precede later accepted same-key output,
// including prequeued commands. Immediate rejections and cancel replies are not
// fenced.
ET_EXPERIMENTAL int run_multiplexed_worker(
    extension::llm::serving::ServingRuntime& runtime,
    int input_fd,
    int output_fd,
    MultiplexedWorkerConfig config = {});

} // namespace executorch::examples::llm_server

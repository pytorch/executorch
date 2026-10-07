/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/examples/models/muse-glimmer/runtime/runners/batching_bootstrap.h>
#include <unistd.h>
#include <csignal>
#include <iostream>

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // Keep model diagnostics out of the JSONL protocol stream.
  const int output = dup(STDOUT_FILENO);
  if (output < 0)
    return 1;
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    close(output);
    return 1;
  }
  std::signal(SIGPIPE, SIG_IGN);
  int status = 1;
  try {
    namespace llm = executorch::extension::llm;
    auto batching_runtime = llm::create_muse_glimmer_batching_runtime();
    executorch::examples::llm_server::MultiplexedWorkerConfig transport;
    transport.max_inflight_requests = FLAGS_max_inflight_requests;
    transport.max_input_frame_bytes = FLAGS_max_input_frame_bytes;
    transport.max_frame_bytes = 1024 * 1024;
    transport.prompt_preparer =
        [spec = batching_runtime->backend.preparation](
            const nlohmann::json& request,
            const llm::serving::PromptPreparationContext& context) {
          return llm::prepare_muse_glimmer_prompt(request, context, spec);
        };
    status = executorch::examples::llm_server::run_multiplexed_worker(
        *batching_runtime->runtime, STDIN_FILENO, output, std::move(transport));
  } catch (const std::exception& error) {
    std::cerr << "batching worker failed: " << error.what() << '\n';
  }
  close(output);
  return status;
}

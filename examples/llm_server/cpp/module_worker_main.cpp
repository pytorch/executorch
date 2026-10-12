/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unistd.h>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include <gflags/gflags.h>

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/module_executor.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/module/module.h>

DEFINE_string(pte, "", "Packed off-graph-cache text model program");
DEFINE_string(tokenizer, "", "Tokenizer matching the program");
DEFINE_int32(max_sessions, 4, "Maximum logical serving sessions");
DEFINE_int32(
    prefix_cache_entries,
    0,
    "Greedy creation-only prompt snapshots; zero disables prefix caching");
DEFINE_int32(max_session_tokens, 2048, "Maximum retained tokens per session");
DEFINE_int32(max_decode_sequences, 2, "Decode slots in one packed forward");
DEFINE_int32(
    max_inflight_requests,
    64,
    "Bounded wire generation/lifecycle capacity");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_pte.empty() || FLAGS_tokenizer.empty() || FLAGS_max_sessions <= 0 ||
      FLAGS_max_session_tokens <= 0 || FLAGS_max_decode_sequences <= 0 ||
      FLAGS_max_inflight_requests <= 0 || FLAGS_prefix_cache_entries < 0) {
    std::cerr
        << "--pte and --tokenizer are required; capacity limits must be positive "
           "and prefix_cache_entries nonnegative\n";
    return 1;
  }
  const auto physical_sessions = static_cast<std::int64_t>(FLAGS_max_sessions) +
      FLAGS_prefix_cache_entries + (FLAGS_prefix_cache_entries > 0 ? 1 : 0);
  if (physical_sessions > std::numeric_limits<int>::max()) {
    std::cerr
        << "session and prefix-cache capacity exceeds executor row limit\n";
    return 1;
  }
  // Model/backend diagnostics must not corrupt the JSONL output stream.
  const int protocol_output = dup(STDOUT_FILENO);
  if (protocol_output < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    if (protocol_output >= 0)
      close(protocol_output);
    return 1;
  }
  signal(SIGPIPE, SIG_IGN);
  int result = 1;
  try {
    namespace llm = executorch::extension::llm;
    auto tokenizer = llm::load_tokenizer(FLAGS_tokenizer);
    auto module = std::make_unique<executorch::extension::Module>(FLAGS_pte);
    if (!tokenizer || module->load() != executorch::runtime::Error::Ok) {
      throw std::runtime_error("could not load program or tokenizer");
    }
    // Load PROGRAM only. ModuleExecutor binds cache before loading forward.
    const auto dtype = llm::read_activation_dtype(*module);
    const auto context = llm::read_max_context_length(*module);
    if (!dtype.ok() || !context.ok() || FLAGS_max_session_tokens > *context) {
      throw std::runtime_error(
          "invalid metadata or max_session_tokens exceeds model context");
    }
    const auto eos = llm::get_eos_ids(tokenizer.get(), module.get());
    auto executor = llm::batching::ModuleExecutor::create(
        std::move(module),
        static_cast<int>(physical_sessions),
        FLAGS_max_session_tokens,
        static_cast<int>(*dtype),
        -1,
        llm::cache::kind::kBatched);
    if (!executor.ok())
      throw std::runtime_error("could not create packed ModuleExecutor");
    const auto width = (*executor)->preferred_batch_tokens();
    const auto decode_slots =
        static_cast<std::size_t>(FLAGS_max_decode_sequences);
    if (width == 0 || decode_slots >= width) {
      throw std::runtime_error(
          "packed forward width must exceed max_decode_sequences");
    }
    auto scheduler = llm::batching::DecodeFirstScheduler::create(
        width, decode_slots, width - decode_slots);
    if (!scheduler)
      throw std::runtime_error("invalid scheduler limits");
    llm::serving::ServingRuntimeConfig config;
    config.max_sessions = FLAGS_max_sessions;
    config.prefix_cache_capacity = FLAGS_prefix_cache_entries;
    config.max_context_length = FLAGS_max_session_tokens;
    config.max_requests = FLAGS_max_inflight_requests;
    config.max_pending_operations = FLAGS_max_inflight_requests;
    config.default_stop_tokens.assign(eos.begin(), eos.end());
    llm::serving::ServingRuntime runtime(
        **executor, std::move(scheduler), *tokenizer, config);
    executorch::examples::llm_server::MultiplexedWorkerConfig transport;
    transport.max_inflight_requests = FLAGS_max_inflight_requests;
    result = executorch::examples::llm_server::run_multiplexed_worker(
        runtime, STDIN_FILENO, protocol_output, transport);
  } catch (const std::exception& error) {
    std::cerr << "worker startup failed: " << error.what() << '\n';
  }
  close(protocol_output);
  return result;
}

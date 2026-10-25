/* Copyright (c) Meta Platforms, Inc. and affiliates.
 * This source code is licensed under the BSD-style license in the root LICENSE.
 */
#include <executorch/backends/cuda/batching/cuda_executor.h>
#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <gflags/gflags.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <unistd.h>
#include <csignal>
#include <iostream>
#include <limits>
#include <stdexcept>

DEFINE_string(pte, "", "Gemma 4 offgraph sequence program");
DEFINE_string(data_path, "", "External CUDA weights");
DEFINE_string(tokenizer, "", "Canonical tokenizer directory");
DEFINE_int32(max_sessions, 16, "Logical sessions");
DEFINE_int32(max_session_tokens, 131072, "Per-sequence context");
DEFINE_int32(initial_kv_capacity, -1, "Initial CUDA KV capacity; -1 uses default");
DEFINE_int32(max_decode_sequences, 16, "Decode batch limit");
DEFINE_int32(max_inflight_requests, 16, "Admission limit");
DEFINE_int32(prefix_cache_entries, 16, "Prefix snapshots");
DEFINE_int32(prefix_checkpoint_interval, 0, "Periodic prefix checkpoint tokens; zero disables periodic capture");
DEFINE_bool(cuda_graphs, true, "Capture existing static methods into CUDA graphs");
DEFINE_int32(idle_kv_compaction_ms, 250, "Compact runtime physical KV columns after engine idle");
DEFINE_int32(idle_kv_spare_columns, 1, "Spare physical columns after idle compaction");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  const int output = dup(STDOUT_FILENO);
  if (output < 0) return 1;
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    close(output);
    return 1;
  }
  std::signal(SIGPIPE, SIG_IGN);
  int status = 1;
  try {
    namespace llm = executorch::extension::llm;
    namespace cuda = executorch::backends::cuda::batching;
    const int64_t physical = static_cast<int64_t>(FLAGS_max_sessions) +
        FLAGS_prefix_cache_entries + (FLAGS_prefix_cache_entries > 0 ? 1 : 0);
    if (FLAGS_pte.empty() || FLAGS_data_path.empty() || FLAGS_tokenizer.empty() ||
        FLAGS_max_sessions <= 0 || FLAGS_max_session_tokens <= 1 ||
        (FLAGS_initial_kv_capacity != -1 &&
         (FLAGS_initial_kv_capacity <= 0 ||
          FLAGS_initial_kv_capacity > FLAGS_max_session_tokens)) ||
        FLAGS_max_decode_sequences <= 0 || FLAGS_max_inflight_requests <= 0 ||
        FLAGS_prefix_cache_entries < 0 || FLAGS_prefix_checkpoint_interval < 0 ||
        FLAGS_idle_kv_compaction_ms < 0 || FLAGS_idle_kv_spare_columns < 0 ||
        (FLAGS_prefix_checkpoint_interval > 0 && FLAGS_prefix_cache_entries == 0) || physical > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("Invalid model paths or batching limits");
    }
    tokenizers::HFTokenizer tokenizer;
    if (tokenizer.load(FLAGS_tokenizer) != tokenizers::Error::Ok) {
      throw std::runtime_error("Could not load canonical Gemma tokenizer");
    }
    cuda::CudaExecutorConfig executor_config;
    executor_config.model_path = FLAGS_pte;
    executor_config.data_files = {FLAGS_data_path};
    executor_config.max_sessions = static_cast<int>(physical);
    executor_config.max_session_tokens = FLAGS_max_session_tokens;
    executor_config.initial_capacity = FLAGS_initial_kv_capacity;
    executor_config.options.cuda_graph_for_static_methods = FLAGS_cuda_graphs;
    executor_config.idle_kv_compaction_ms = FLAGS_idle_kv_compaction_ms;
    executor_config.idle_kv_spare_columns = FLAGS_idle_kv_spare_columns;
    auto created = cuda::CudaExecutor::create(executor_config);
    if (!created.ok()) {
      throw std::runtime_error("Could not create Gemma CUDA executor: " +
          std::to_string(static_cast<int>(created.error())));
    }
    auto executor = std::move(*created);
    const size_t budget = executor->preferred_batch_tokens();
    if (budget <= static_cast<size_t>(FLAGS_max_decode_sequences)) {
      throw std::invalid_argument("Forward budget must leave room for prefill");
    }
    auto scheduler = llm::batching::DecodeFirstScheduler::create(
        budget, FLAGS_max_decode_sequences, budget - FLAGS_max_decode_sequences);
    if (!scheduler) throw std::runtime_error("Invalid scheduler limits");
    llm::serving::ServingRuntimeConfig config;
    config.max_sessions = FLAGS_max_sessions;
    config.max_context_length = executor->max_session_tokens();
    config.max_requests = FLAGS_max_inflight_requests;
    config.max_pending_operations = FLAGS_max_inflight_requests;
    config.prefix_cache_capacity = FLAGS_prefix_cache_entries;
    config.prefix_checkpoint_interval = FLAGS_prefix_checkpoint_interval;
    config.default_stop_tokens = {1, 106, 50};
    llm::serving::ServingRuntime runtime(*executor, std::move(scheduler), tokenizer, config);
    executorch::examples::llm_server::MultiplexedWorkerConfig transport;
    transport.max_inflight_requests = FLAGS_max_inflight_requests;
    transport.max_input_frame_bytes = 32 * 1024 * 1024;
    transport.max_frame_bytes = 1024 * 1024;
    status = executorch::examples::llm_server::run_multiplexed_worker(
        runtime, STDIN_FILENO, output, std::move(transport));
  } catch (const std::exception& error) {
    std::cerr << "Gemma batching worker failed: " << error.what() << '\n';
  }
  close(output);
  return status;
}

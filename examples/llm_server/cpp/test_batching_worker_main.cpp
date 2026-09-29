/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/llm_server/cpp/multiplexed_worker.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

#include <pytorch/tokenizers/tokenizer.h>

#include <unistd.h>
#include <chrono>
#include <csignal>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

namespace batching = executorch::extension::llm::batching;
namespace serving = executorch::extension::llm::serving;

namespace {

// Test-only gate: returning no work lets Runner process further submissions.
// Waiting inside get_work() would prevent the second session from arriving.
class InitialBatchGate final : public batching::Scheduler {
 public:
  explicit InitialBatchGate(bool gate)
      : inner_(batching::DecodeFirstScheduler::create(512, 8, 256)),
        released_(!gate) {}

  bool submit(std::vector<batching::Task> tasks) override {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<batching::SessionId> ids;
    if (!released_) {
      for (const auto& task : tasks) {
        ids.insert(task.input.sid);
      }
    }
    if (!inner_->submit(std::move(tasks))) {
      return false;
    }
    if (!released_) {
      queued_.insert(ids.begin(), ids.end());
      if (queued_.size() >= 2) {
        released_ = true;
        queued_.clear();
      }
    }
    return true;
  }

  bool has_work() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_ && inner_->has_work();
  }

  std::vector<batching::Task> get_work() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return released_ ? inner_->get_work() : std::vector<batching::Task>{};
  }

  std::vector<batching::Task> cancel(batching::SessionId sid) override {
    std::lock_guard<std::mutex> lock(mutex_);
    queued_.erase(sid);
    return inner_->cancel(sid);
  }

  std::vector<batching::Task> clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    queued_.clear();
    return inner_->clear();
  }

  std::size_t max_prefill_chunk_size() const override {
    return inner_->max_prefill_chunk_size();
  }

 private:
  std::unique_ptr<batching::DecodeFirstScheduler> inner_;
  mutable std::mutex mutex_;
  std::set<batching::SessionId> queued_;
  bool released_;
};

class TraceExecutor final : public batching::testing::FakeExecutor {
 public:
  explicit TraceExecutor(std::ostream* trace) : trace_(trace) {}

  std::optional<batching::SessionId> open_session() override {
    auto session = FakeExecutor::open_session();
    if (session) {
      history_.emplace(*session, std::vector<batching::Token>{});
    }
    return session;
  }

  void close_session(batching::SessionId session) override {
    history_.erase(session);
    FakeExecutor::close_session(session);
  }

  std::optional<batching::SessionId> clone(
      batching::SessionId source,
      batching::Position upto) override {
    const auto it = history_.find(source);
    if (it == history_.end() || upto < 0 ||
        static_cast<std::size_t>(upto) > it->second.size()) {
      return std::nullopt;
    }
    auto session = FakeExecutor::open_session();
    if (session) {
      history_.emplace(
          *session,
          std::vector<batching::Token>(
              it->second.begin(), it->second.begin() + upto));
    }
    return session;
  }

  bool execute(const batching::BatchInput& batch, batching::BatchOutput& output)
      override {
    if (trace_) {
      std::set<batching::SessionId> ids;
      for (const auto& input : batch.inputs) {
        ids.insert(input.sid);
      }
      *trace_ << ids.size() << '\n';
      trace_->flush();
    }
    // Give subprocess readers time to exercise cancellation/backpressure.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!FakeExecutor::execute(batch, output)) {
      return false;
    }
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      const auto& input = batch.inputs[i];
      const auto it = history_.find(input.sid);
      const auto position =
          static_cast<std::size_t>(input.position) + input.offset;
      if (it == history_.end() || position > it->second.size()) {
        return false;
      }
      auto& tokens = it->second;
      tokens.resize(position);
      tokens.insert(
          tokens.end(),
          input.tokens->begin() + input.offset,
          input.tokens->begin() + input.offset + input.size);
      if (output.outputs[i] && !output.outputs[i]->tokens.empty()) {
        const auto& generated = output.outputs[i]->tokens;
        tokens.insert(tokens.end(), generated.begin(), generated.end() - 1);
      }
    }
    return true;
  }

 private:
  // Engine-thread-only committed history; the final prediction stays pending.
  std::map<batching::SessionId, std::vector<batching::Token>> history_;
  std::ostream* trace_;
};

class ByteTokenizer final : public tokenizers::Tokenizer {
 public:
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }

  tokenizers::Result<std::vector<uint64_t>>
  encode(const std::string& text, int8_t, int8_t) const override {
    std::vector<uint64_t> ids;
    for (const unsigned char byte : text) {
      ids.push_back(byte);
    }
    return ids;
  }

  tokenizers::Result<std::string> decode(uint64_t, uint64_t token, bool)
      const override {
    return std::string(1, static_cast<char>('a' + token % 26));
  }

  tokenizers::Result<std::string> id_to_piece(uint64_t token) const override {
    return decode(0, token, false);
  }

  tokenizers::Result<uint64_t> piece_to_id(const std::string&) const override {
    return tokenizers::Error::Internal;
  }
};

} // namespace

int main(int argc, char** argv) {
  bool gate = false;
  bool stop_immediately = false;
  bool prefix_cache = false;
  std::string trace_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--gate-two") {
      gate = true;
    } else if (arg == "--stop-immediately") {
      stop_immediately = true;
    } else if (arg == "--prefix-cache") {
      prefix_cache = true;
    } else if (arg == "--trace" && i + 1 < argc) {
      trace_path = argv[++i];
    } else {
      return 2;
    }
  }
  std::ofstream trace;
  if (!trace_path.empty()) {
    trace.open(trace_path);
    if (!trace) {
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  TraceExecutor executor(trace.is_open() ? &trace : nullptr);
  if (stop_immediately) {
    executor.stop_token = 9;
  }
  ByteTokenizer tokenizer;
  serving::ServingRuntimeConfig config;
  config.max_sessions = 8;
  config.max_context_length = 4096;
  config.prefix_cache_capacity = prefix_cache ? 2 : 0;
  executor.capacity = static_cast<int>(
      config.max_sessions +
      (prefix_cache ? config.prefix_cache_capacity + 1 : 0));
  if (stop_immediately) {
    config.default_stop_tokens = {9};
  }
  serving::ServingRuntime runtime(
      executor, std::make_unique<InitialBatchGate>(gate), tokenizer, config);
  executorch::examples::llm_server::MultiplexedWorkerConfig worker_config;
  worker_config.max_inflight_requests = 16;
  return executorch::examples::llm_server::run_multiplexed_worker(
      runtime, STDIN_FILENO, STDOUT_FILENO, worker_config);
}

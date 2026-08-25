/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <vector>

#include <executorch/extension/llm/batching/executor.h>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {
namespace testing {

// Scriptable executor for runner tests. It records the slices it was handed
// and answers with deterministic tokens.
//
// The locking here is not the Executor contract's. executor.h promises every
// call arrives on the runner's engine thread, so a real implementation needs
// none. It guards the test-facing surface instead, which the test thread
// touches while the engine thread is driving the fake: the seen(), opened(),
// and open_count() observers, and the hold()/release() gate that parks the
// engine inside execute() so a test can land a race deterministically.
class FakeExecutor : public Executor {
 public:
  struct Seen {
    SessionId session;
    Position position;
    std::size_t offset;
    std::size_t size;
    std::optional<std::uint64_t> sampling_seed;

    Position effective_position() const {
      return position + static_cast<Position>(offset);
    }
  };

  std::optional<SessionId> open_session() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (static_cast<int>(open_.size()) >= capacity) {
      return std::nullopt;
    }
    const SessionId sid = next_session_++;
    open_.insert(sid);
    opened_.push_back(sid);
    return sid;
  }

  void close_session(SessionId session) override {
    std::lock_guard<std::mutex> lock(mutex_);
    open_.erase(session);
    sampling_.erase(session);
    closed_.push_back(session);
  }

  void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) override {
    std::lock_guard<std::mutex> lock(mutex_);
    sampling_.insert_or_assign(session, SamplingState(params, seed));
  }

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    {
      std::unique_lock<std::mutex> lock(gate_mutex_);
      in_execute_.store(true);
      gate_cv_.wait(lock, [this] { return !held_; });
      in_execute_.store(false);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    batch_sizes_.push_back(static_cast<int>(batch.inputs.size()));
    for (const Input& input : batch.inputs) {
      if (sampling_.count(input.sid) == 0) {
        executed_without_sampling_state_ = true;
      }
      const auto sampling = sampling_.find(input.sid);
      seen_.push_back(Seen{
          input.sid,
          input.position,
          input.offset,
          input.size,
          sampling == sampling_.end() ? std::nullopt : sampling->second.seed});
    }
    if (batches_++ >= fail_batches_from && fail_batches_from >= 0) {
      return false;
    }

    out.outputs.clear();
    out.outputs.reserve(batch.inputs.size());
    for (const Input& input : batch.inputs) {
      if (!input.produce_output) {
        out.outputs.emplace_back(std::nullopt);
        continue;
      }

      Output output;
      output.sid = wrong_sid ? input.sid + 1000 : input.sid;
      output.tokens = produce(input);
      if (empty_tokens) {
        output.tokens.clear();
      }
      out.outputs.emplace_back(std::move(output));
    }
    return true;
  }

  int capacity = 8;
  // Batch index from which execute() starts failing. Negative never fails.
  int fail_batches_from = -1;
  // Once a session has produced emit_before_stop tokens, every later one is
  // stop_token. Counted per session across the whole run, so a stop can be
  // placed part way into a multi-token decode. A negative stop_token disables
  // this.
  Token stop_token = -1;
  int emit_before_stop = 0;
  // Tokens a decode step produces. 1 is a plain executor; more simulates a
  // speculative one answering with the run it accepted plus the model's own
  // next token. Prefill always produces one whatever this is.
  std::size_t tokens_per_decode = 1;
  // Malformed answers. An Output carries only the tokens an input produced, so
  // the only ways to break the contract are to produce none, or to answer for
  // a session the input did not name.
  bool empty_tokens = false;
  bool wrong_sid = false;

  void hold() {
    std::lock_guard<std::mutex> lock(gate_mutex_);
    held_ = true;
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(gate_mutex_);
      held_ = false;
    }
    gate_cv_.notify_all();
  }

  bool in_execute() const {
    return in_execute_.load();
  }

  std::vector<Seen> seen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_;
  }

  std::vector<int> batch_sizes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return batch_sizes_;
  }

  std::vector<SessionId> opened() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return opened_;
  }

  std::vector<SessionId> closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

  bool has_sampling_state(SessionId session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sampling_.count(session) != 0;
  }

  std::optional<std::uint64_t> sampling_seed(SessionId session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sampling_.find(session);
    return it == sampling_.end() ? std::nullopt : it->second.seed;
  }

  // The policy installed for the session, so tests can check it arrived.
  std::optional<SamplingParams> sampling_params(SessionId session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sampling_.find(session);
    return it == sampling_.end()
        ? std::nullopt
        : std::optional<SamplingParams>(it->second.params);
  }

  bool executed_without_sampling_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return executed_without_sampling_state_;
  }

  int open_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(open_.size());
  }

 private:
  // Decode is inferred from a single-token input, since Input does not carry
  // Task::is_decode. Good enough for a fake: the runner only ever feeds one
  // token to continue.
  std::vector<Token> produce(const Input& input) {
    const std::size_t n = input.size == 1 ? tokens_per_decode : 1;
    std::vector<Token> produced;
    produced.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      produced.push_back(next_token(input.sid));
    }
    return produced;
  }

  Token next_token(SessionId session) {
    const int n = ++produced_[session];
    if (stop_token >= 0 && n > emit_before_stop) {
      return stop_token;
    }
    auto it = sampling_.find(session);
    if (it == sampling_.end()) {
      return session * 1000 + n;
    }
    return session * 1000 + static_cast<Token>(it->second.rng() % 1000);
  }

  mutable std::mutex mutex_;

  struct SamplingState {
    SamplingState(
        const SamplingParams& requested_params,
        std::optional<std::uint64_t> requested_seed)
        : params(requested_params),
          seed(requested_seed),
          rng(requested_seed.value_or(random_seed())) {}

    static std::uint64_t random_seed() {
      std::random_device random;
      return (static_cast<std::uint64_t>(random()) << 32) ^ random();
    }

    SamplingParams params;
    std::optional<std::uint64_t> seed;
    std::mt19937_64 rng;
  };

  std::mutex gate_mutex_;
  std::condition_variable gate_cv_;
  bool held_ = false;
  std::atomic<bool> in_execute_{false};
  SessionId next_session_ = 1;
  std::set<SessionId> open_;
  std::vector<SessionId> opened_;
  std::vector<SessionId> closed_;
  std::map<SessionId, SamplingState> sampling_;
  bool executed_without_sampling_state_ = false;
  std::vector<Seen> seen_;
  std::vector<int> batch_sizes_;
  std::map<SessionId, int> produced_;
  int batches_ = 0;
};

} // namespace testing
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch

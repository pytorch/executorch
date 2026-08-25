/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
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

// Scriptable executor for runner tests. It records positioned slices and
// returns deterministic tokens plus an explicitly positioned continuation.
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
    return sid;
  }

  void close_session(SessionId session) override {
    std::lock_guard<std::mutex> lock(mutex_);
    open_.erase(session);
    sampling_.erase(session);
    closed_.push_back(session);
  }

  void set_sampling_seed(
      SessionId session,
      std::optional<std::uint64_t> seed) override {
    std::lock_guard<std::mutex> lock(mutex_);
    sampling_.insert_or_assign(session, SamplingState(seed));
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

      std::vector<Token> produced = produce(input);
      Output output;
      output.sid = input.sid;
      output.tokens =
          std::make_shared<const std::vector<Token>>(std::move(produced));
      if (!omit_continuation && !output.tokens->empty()) {
        const Position next_position =
            input.position + static_cast<Position>(input.offset) +
            static_cast<Position>(input.size) +
            static_cast<Position>(output.tokens->size()) - 1;
        auto next_tokens = std::make_shared<const std::vector<Token>>(
            std::vector<Token>{output.tokens->back()});
        if (empty_continuation) {
          next_tokens = std::make_shared<const std::vector<Token>>();
        }
        output.next = Output::Continuation{
            null_continuation_tokens ? nullptr : std::move(next_tokens),
            continuation_position.value_or(next_position)};
      }
      out.outputs.emplace_back(std::move(output));
    }
    return true;
  }

  int capacity = 8;
  int fail_batches_from = -1;
  Token stop_token = -1;
  int emit_before_stop = 0;
  bool omit_continuation = false;
  bool null_continuation_tokens = false;
  bool empty_continuation = false;
  std::optional<Position> continuation_position;

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

  bool executed_without_sampling_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return executed_without_sampling_state_;
  }

  int open_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(open_.size());
  }

 protected:
  virtual std::vector<Token> produce(const Input& input) {
    return {next_token(input.sid)};
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

 private:
  struct SamplingState {
    explicit SamplingState(std::optional<std::uint64_t> requested_seed)
        : seed(requested_seed), rng(requested_seed.value_or(random_seed())) {}

    static std::uint64_t random_seed() {
      std::random_device random;
      return (static_cast<std::uint64_t>(random()) << 32) ^ random();
    }

    std::optional<std::uint64_t> seed;
    std::mt19937_64 rng;
  };

  std::mutex gate_mutex_;
  std::condition_variable gate_cv_;
  bool held_ = false;
  std::atomic<bool> in_execute_{false};
  SessionId next_session_ = 1;
  std::set<SessionId> open_;
  std::vector<SessionId> closed_;
  std::map<SessionId, SamplingState> sampling_;
  bool executed_without_sampling_state_ = false;
  std::vector<Seen> seen_;
  std::vector<int> batch_sizes_;
  std::map<SessionId, int> produced_;
  int batches_ = 0;
};

class FakeDFlashExecutor : public FakeExecutor {
 public:
  std::int32_t n_draft = 4;
  std::vector<std::int32_t> acceptance;

 protected:
  std::vector<Token> produce(const Input& input) override {
    if (input.size != 1) {
      return {next_token(input.sid)};
    }
    const std::int32_t accepted = accepted_for(input.sid);
    std::vector<Token> tokens;
    tokens.reserve(static_cast<std::size_t>(accepted) + 1);
    for (std::int32_t i = 0; i <= accepted; ++i) {
      tokens.push_back(next_token(input.sid));
    }
    return tokens;
  }

 private:
  std::int32_t accepted_for(SessionId session) {
    if (acceptance.empty()) {
      return n_draft;
    }
    const std::size_t round =
        static_cast<std::size_t>(rounds_[session]++) % acceptance.size();
    return std::min(acceptance[round], n_draft);
  }

  std::map<SessionId, int> rounds_;
};

} // namespace testing
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch

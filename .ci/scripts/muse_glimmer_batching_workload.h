/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/serving/serving_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

// Private CI workload control, shared by correctness and measurement tools.
namespace muse_glimmer_ci {
namespace batching = ::executorch::extension::llm::batching;
namespace serving = ::executorch::extension::llm::serving;

struct RequestOptions {
  std::int32_t requests = 8;
  std::int32_t prompt_tokens = 512;
  std::int32_t max_new_tokens = 128;
  std::int32_t stagger_ms = 0;
  std::int32_t max_decode = 32;
  std::int32_t timeout_s = 600;
};

inline void check_session_capacity(
    std::size_t required,
    std::size_t available) {
  if (available == 0 || required == 0 || required > available) {
    throw std::invalid_argument("workload exceeds --max_sessions");
  }
}

inline void validate_requests(
    const RequestOptions& options,
    std::size_t max_sessions,
    std::size_t max_context_length = 0) {
  if (options.requests <= 0 || options.prompt_tokens <= 0 ||
      options.max_new_tokens <= 0 || options.stagger_ms < 0 ||
      options.max_decode <= 0 || options.timeout_s <= 0) {
    throw std::invalid_argument("invalid request workload options");
  }
  check_session_capacity(options.requests, max_sessions);
  const auto tokens = static_cast<std::uint64_t>(options.prompt_tokens) +
      static_cast<std::uint64_t>(options.max_new_tokens);
  if (max_context_length > 0 && tokens > max_context_length) {
    throw std::invalid_argument("request token budget exceeds context");
  }
}

inline std::vector<batching::Token> prompt_ids(
    std::size_t length,
    std::size_t salt) {
  std::vector<batching::Token> ids(length);
  for (std::size_t i = 0; i < length; ++i) {
    ids[i] = 1000 + (i * 7919 + salt * 104729) % 99000;
  }
  return ids;
}

struct BatchEvidence {
  std::size_t max_sessions = 0;
  std::size_t multi_session_steps = 0;
  std::size_t mixed_steps = 0;

  void observe(const batching::BatchInput& batch) {
    std::set<batching::SessionId> sessions;
    std::size_t decodes = 0;
    for (const auto& input : batch.inputs) {
      sessions.insert(input.sid);
      decodes += input.size == 1 && input.produce_output &&
          std::holds_alternative<batching::TokenInputPtr>(input.payload);
    }
    max_sessions = std::max(max_sessions, sessions.size());
    if (sessions.size() > 1) {
      ++multi_session_steps;
      if (decodes > 0 && batch.size() > decodes) {
        ++mixed_steps;
      }
    }
  }
};

// Counts distinct owners only after the forwarded executor call succeeds.
class EvidenceExecutor final : public batching::Executor {
 public:
  explicit EvidenceExecutor(batching::Executor& inner) : inner_(inner) {}

  bool initialize() override {
    return inner_.initialize();
  }
  std::size_t preferred_batch_tokens() const override {
    return inner_.preferred_batch_tokens();
  }
  std::optional<batching::SessionId> open_session() override {
    return inner_.open_session();
  }
  void close_session(batching::SessionId session) override {
    inner_.close_session(session);
  }
  std::optional<batching::SessionId> clone(
      batching::SessionId source,
      batching::Position upto) override {
    return inner_.clone(source, upto);
  }
  void set_sampling(
      batching::SessionId session,
      const batching::SamplingParams& params,
      std::optional<std::uint64_t> seed) override {
    inner_.set_sampling(session, params, seed);
  }
  bool accepts(const batching::PreparedInput& input) const override {
    return inner_.accepts(input);
  }
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& out)
      override {
    if (!inner_.execute(batch, out)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    evidence_.observe(batch);
    return true;
  }
  BatchEvidence evidence() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evidence_;
  }

 private:
  batching::Executor& inner_;
  mutable std::mutex mutex_;
  BatchEvidence evidence_;
};

inline void check_batching(const BatchEvidence& evidence) {
  if (evidence.max_sessions <= 1 || evidence.multi_session_steps == 0) {
    throw std::runtime_error("no successful multi-session executor step");
  }
}

inline std::unique_ptr<serving::ServingRuntime> make_runtime(
    batching::Executor& executor,
    std::unique_ptr<batching::Scheduler> scheduler,
    const tokenizers::Tokenizer& tokenizer,
    serving::ServingRuntimeConfig config,
    serving::ModelPreparer preparer) {
  if (!scheduler) {
    throw std::invalid_argument("unusable scheduler limits");
  }
  return std::make_unique<serving::ServingRuntime>(
      executor, std::move(scheduler), tokenizer, config, std::move(preparer));
}

struct Inflight {
  std::mutex mutex;
  std::condition_variable changed;
  std::size_t running = 0;
  std::size_t failed = 0;
  std::string error;

  void finished(const serving::TerminalEvent& terminal) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      --running;
      if (terminal.finish_reason == serving::FinishReason::Failed) {
        ++failed;
        if (terminal.error && error.empty()) {
          error = terminal.error->message;
        }
      }
    }
    changed.notify_all();
  }
};

inline std::optional<serving::RequestHandle> submit_request(
    serving::ServingRuntime& runtime,
    Inflight& inflight,
    std::size_t length,
    std::size_t salt,
    std::int32_t new_tokens,
    std::function<void(const serving::TerminalEvent&)> on_terminal = {},
    std::function<void()> on_text = {}) {
  if (length == 0 || new_tokens <= 0) {
    throw std::invalid_argument("invalid prompt or generation budget");
  }
  serving::PromptInput prompt;
  // Model preparation adds BOS, which is included in the requested length.
  prompt.segments.emplace_back(prompt_ids(length - 1, salt));
  serving::GenerationOptions options;
  options.max_new_tokens = new_tokens;
  options.sampling.temperature = 0.0f;
  options.seed = salt;
  {
    std::lock_guard<std::mutex> lock(inflight.mutex);
    ++inflight.running;
  }
  auto result = runtime.generate(
      std::nullopt,
      std::move(prompt),
      options,
      [&inflight, on_terminal, on_text](serving::GenerationEvent event) {
        if (const auto* text = std::get_if<serving::TextEvent>(&event);
            text && !text->text.empty() && on_text) {
          on_text();
        }
        if (const auto* terminal =
                std::get_if<serving::TerminalEvent>(&event)) {
          if (on_terminal) {
            on_terminal(*terminal);
          }
          inflight.finished(*terminal);
        }
      });
  if (const auto* error = std::get_if<serving::ServingError>(&result)) {
    std::lock_guard<std::mutex> lock(inflight.mutex);
    --inflight.running;
    ++inflight.failed;
    inflight.error = error->message;
    return std::nullopt;
  }
  return std::get<serving::RequestHandle>(std::move(result));
}

struct RequestResult {
  std::size_t accepted = 0;
  std::vector<std::optional<serving::TerminalEvent>> terminals;
};

inline void check_requests_complete(
    const RequestOptions& options,
    const RequestResult& result) {
  if (result.accepted != static_cast<std::size_t>(options.requests) ||
      result.terminals.size() != result.accepted) {
    throw std::runtime_error("not all workload requests were accepted");
  }
  const auto incomplete =
      [&](const std::optional<serving::TerminalEvent>& terminal) {
        return !terminal ||
            terminal->finish_reason != serving::FinishReason::Length ||
            terminal->stats.prompt_tokens !=
            static_cast<std::size_t>(options.prompt_tokens) ||
            terminal->stats.completion_tokens !=
            static_cast<std::size_t>(options.max_new_tokens);
      };
  if (std::any_of(
          result.terminals.begin(), result.terminals.end(), incomplete)) {
    throw std::runtime_error("request failed or token budget incomplete");
  }
}

// Submission runs on the caller; text/terminal hooks run on delivery. Captured
// state must outlive run_requests. Hooks must be bounded and must not throw.
struct RequestCallbacks {
  std::function<void(std::size_t)> submitted;
  std::function<void(std::size_t)> text;
  std::function<void(std::size_t, const serving::TerminalEvent&)> terminal;
};

inline RequestResult run_requests(
    const RequestOptions& options,
    batching::Executor& executor,
    const tokenizers::Tokenizer& tokenizer,
    serving::ServingRuntimeConfig config,
    serving::ModelPreparer preparer,
    const RequestCallbacks& callbacks = {}) {
  validate_requests(options, config.max_sessions, config.max_context_length);
  check_session_capacity(options.requests, config.max_requests);
  const auto decodes =
      static_cast<std::size_t>(std::min(options.max_decode, options.requests));
  const auto budget = std::max(executor.preferred_batch_tokens(), decodes + 1);
  auto scheduler =
      batching::DecodeFirstScheduler::create(budget, decodes, budget - decodes);
  config.default_stop_tokens = {};
  RequestResult result;
  result.terminals.resize(options.requests);
  Inflight inflight;
  std::vector<serving::RequestHandle> handles;
  // Runtime destruction fences callbacks before their captured state dies.
  auto runtime = make_runtime(
      executor, std::move(scheduler), tokenizer, config, std::move(preparer));
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::seconds(options.timeout_s);
  bool timed_out = false;
  for (std::size_t i = 0; i < result.terminals.size(); ++i) {
    if (i > 0 && options.stagger_ms > 0) {
      std::this_thread::sleep_until(std::min(
          deadline,
          Clock::now() + std::chrono::milliseconds(options.stagger_ms)));
    }
    if (Clock::now() >= deadline) {
      timed_out = true;
      break;
    }
    if (callbacks.submitted) {
      callbacks.submitted(i);
    }
    auto handle = submit_request(
        *runtime,
        inflight,
        options.prompt_tokens,
        i,
        options.max_new_tokens,
        [&result, &callbacks, i](const serving::TerminalEvent& terminal) {
          result.terminals[i] = terminal;
          if (callbacks.terminal) {
            callbacks.terminal(i, terminal);
          }
        },
        [&callbacks, i] {
          if (callbacks.text) {
            callbacks.text(i);
          }
        });
    if (!handle) {
      break;
    }
    ++result.accepted;
    handles.push_back(std::move(*handle));
  }
  {
    std::unique_lock<std::mutex> lock(inflight.mutex);
    timed_out = timed_out || Clock::now() >= deadline ||
        !inflight.changed.wait_until(lock, deadline, [&] {
          return inflight.running == 0 || inflight.failed > 0;
        });
    if (timed_out || inflight.failed > 0 ||
        result.accepted != result.terminals.size()) {
      lock.unlock();
      for (const auto& handle : handles) {
        handle.cancel();
      }
      runtime->shutdown();
    }
  }
  for (const auto& handle : handles) {
    handle.wait();
  }
  runtime->shutdown();
  if (timed_out) {
    throw std::runtime_error("request workload timed out");
  }
  check_requests_complete(options, result);
  return result;
}

} // namespace muse_glimmer_ci

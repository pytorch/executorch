/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace executorch::extension::llm;
using namespace executorch::extension::llm::serving;
using batching::Token;
using namespace std::chrono_literals;

namespace {

class Gate {
 public:
  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
    entered_ = false;
  }
  void enter() {
    std::unique_lock<std::mutex> lock(mutex_);
    entered_ = true;
    cv_.notify_all();
    cv_.wait(lock, [&] { return !held_; });
  }
  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, 5s, [&] { return entered_; });
  }
  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = false;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool held_ = false;
  bool entered_ = false;
};

class Tokenizer : public tokenizers::Tokenizer {
 public:
  std::map<std::string, std::vector<Token>> encodings{{"hello", {10, 11}}};
  std::map<Token, std::string> pieces{
      {100, "A"},
      {101, "B"},
      {102, "C"},
      {103, "D"}};
  mutable Gate encoding;
  mutable std::atomic<int> encode_calls{0};
  mutable std::thread::id encoder_thread;

  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::vector<Token>>
  encode(const std::string& text, int8_t bos, int8_t eos) const override {
    ++encode_calls;
    encoder_thread = std::this_thread::get_id();
    encoding.enter();
    EXPECT_EQ(bos, 0);
    EXPECT_EQ(eos, 0);
    auto it = encodings.find(text);
    if (it == encodings.end()) {
      return tokenizers::Error::Internal;
    }
    return it->second;
  }
  tokenizers::Result<std::string> decode(Token, Token token, bool)
      const override {
    auto it = pieces.find(token);
    if (it == pieces.end()) {
      return tokenizers::Error::Internal;
    }
    return it->second;
  }
  tokenizers::Result<std::string> id_to_piece(Token) const override {
    return tokenizers::Error::Internal;
  }
  tokenizers::Result<Token> piece_to_id(const std::string&) const override {
    return tokenizers::Error::Internal;
  }
};

class Executor : public batching::testing::FakeExecutor {
 public:
  Gate opening;
  Gate executing;
  Gate second_step;
  std::atomic<bool> refuse_open{false};
  std::atomic<int> clones{0};
  std::atomic<int> steps{0};
  std::size_t burst = 1;
  std::vector<Token> script{100, 101, 102, 103};
  std::vector<std::vector<Token>> fed;
  std::thread::id engine_thread;

  std::optional<batching::SessionId> open_session() override {
    opening.enter();
    if (refuse_open.load()) {
      return std::nullopt;
    }
    return FakeExecutor::open_session();
  }
  std::optional<batching::SessionId> clone(
      batching::SessionId,
      batching::Position) override {
    ++clones;
    return std::nullopt;
  }
  bool execute(const batching::BatchInput& input, batching::BatchOutput& output)
      override {
    engine_thread = std::this_thread::get_id();
    const auto step = ++steps;
    executing.enter();
    if (step >= 2) {
      second_step.enter();
    }
    if (!FakeExecutor::execute(input, output)) {
      return false;
    }
    for (std::size_t i = 0; i < input.inputs.size(); ++i) {
      const auto& slice = input.inputs[i];
      const auto& source = *std::get<batching::TokenInputPtr>(slice.payload);
      fed.emplace_back(
          source.begin() + slice.offset,
          source.begin() + slice.offset + slice.size);
      if (output.outputs[i]) {
        auto& tokens = output.outputs[i]->tokens;
        tokens.clear();
        for (std::size_t j = 0; j < burst; ++j) {
          auto& next = produced_[slice.sid];
          tokens.push_back(script[next++ % script.size()]);
        }
      }
    }
    return true;
  }

 private:
  std::map<batching::SessionId, std::size_t> produced_;
};

struct Events {
  std::string text;
  std::optional<TerminalEvent> terminal;
  std::size_t terminals = 0;
  std::thread::id sink_thread;
  Gate blocked;
  Gate terminal_blocked;
  bool block_first = false;
  bool block_terminal = false;
  std::function<void()> on_text;

  void accept(GenerationEvent event) {
    sink_thread = std::this_thread::get_id();
    if (auto* piece = std::get_if<TextEvent>(&event)) {
      if (block_first) {
        block_first = false;
        blocked.enter();
      }
      if (on_text) {
        on_text();
      }
      text += piece->text;
    } else {
      terminal = std::get<TerminalEvent>(std::move(event));
      ++terminals;
      if (block_terminal) {
        terminal_blocked.enter();
      }
    }
  }
};

PromptInput ids(std::vector<Token> tokens) {
  return PromptInput{{make_token_input(std::move(tokens))}};
}

template <class Predicate>
bool wait_until(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void expect_same_terminal(
    const TerminalEvent& actual,
    const TerminalEvent& expected) {
  EXPECT_EQ(actual.finish_reason, expected.finish_reason);
  ASSERT_EQ(actual.error.has_value(), expected.error.has_value());
  if (expected.error) {
    EXPECT_EQ(actual.error->code, expected.error->code);
    EXPECT_EQ(actual.error->message, expected.error->message);
  }
  EXPECT_EQ(actual.stats.prompt_tokens, expected.stats.prompt_tokens);
  EXPECT_EQ(actual.stats.completion_tokens, expected.stats.completion_tokens);
  EXPECT_EQ(
      actual.stats.reused_prompt_tokens, expected.stats.reused_prompt_tokens);
  EXPECT_EQ(
      actual.stats.prefilled_prompt_tokens,
      expected.stats.prefilled_prompt_tokens);
  EXPECT_EQ(actual.stats.prefill_ms, expected.stats.prefill_ms);
  EXPECT_EQ(actual.stats.decode_ms, expected.stats.decode_ms);
  EXPECT_EQ(actual.stats.total_ms, expected.stats.total_ms);
  EXPECT_EQ(
      actual.stats.session_reset_reason, expected.stats.session_reset_reason);
  EXPECT_EQ(
      actual.stats.generated_token_ids, expected.stats.generated_token_ids);
}

class TextGenerationTest : public ::testing::Test {
 protected:
  Executor executor;
  Tokenizer tokenizer;
  ServingRuntimeConfig config;
  Gate delivery_checkpoint;
  Gate cleanup_checkpoint;
  struct SmallSinkState {
    std::atomic<int> destructions{0};
    std::atomic<int> calls{0};
  } small_sink_state;
  struct InlineSinkState {
    Gate* cleanup = nullptr;
    std::thread::id submitting_thread;
    std::atomic<int> live_owners{0};
    std::atomic<int> copies{0};
    std::atomic<int> original_destructions{0};
    std::atomic<int> copied_destructions{0};
    std::atomic<int> terminals{0};
    std::atomic<bool> cleanup_entered{false};
  } inline_sink_state;
  std::thread submitting_caller;
  std::unique_ptr<ServingRuntime> runtime;
  std::vector<std::shared_ptr<Events>> events;
  std::promise<void> callbacks_finished;
  std::thread callback_watchdog;

  void watch_callbacks() {
    // Bound teardown too if a reentrant runtime call deadlocks.
    callback_watchdog = std::thread([finished =
                                         callbacks_finished.get_future()] {
      if (finished.wait_for(30s) != std::future_status::ready) {
        ADD_FAILURE() << "public terminal callback test or teardown deadlocked";
        std::abort();
      }
    });
  }
  void SetUp() override {
    executorch::runtime::runtime_init();
    config.max_sessions = 4;
    config.max_context_length = 64;
    config.max_requests = 4;
  }
  void start(
      std::size_t chunk = 8,
      std::size_t batch = 32,
      std::size_t decodes = 4) {
    runtime = std::make_unique<ServingRuntime>(
        executor,
        batching::DecodeFirstScheduler::create(batch, decodes, chunk),
        tokenizer,
        config);
  }
  void TearDown() override {
    delivery_checkpoint.release();
    cleanup_checkpoint.release();
    tokenizer.encoding.release();
    executor.opening.release();
    executor.executing.release();
    executor.second_step.release();
    executor.release();
    for (auto& event : events) {
      event->blocked.release();
      event->terminal_blocked.release();
    }
    if (submitting_caller.joinable()) {
      submitting_caller.join();
    }
    runtime.reset();
    EXPECT_EQ(executor.clones.load(), 0);
    callbacks_finished.set_value();
    if (callback_watchdog.joinable()) {
      callback_watchdog.join();
    }
  }
  std::shared_ptr<Events> output() {
    auto event = std::make_shared<Events>();
    events.push_back(event);
    return event;
  }
  RequestHandle submit(
      const std::shared_ptr<Events>& event,
      PromptInput prompt,
      GenerationOptions options = {},
      std::optional<std::string> key = "s") {
    if (!options.max_new_tokens) {
      options.max_new_tokens = 1;
    }
    auto result = runtime->generate(
        std::move(key),
        std::move(prompt),
        std::move(options),
        [event](GenerationEvent update) { event->accept(std::move(update)); });
    if (auto* error = std::get_if<ServingError>(&result)) {
      ADD_FAILURE() << error->message;
      return {};
    }
    return std::get<RequestHandle>(std::move(result));
  }
};

TEST_F(
    TextGenerationTest,
    TextAndMixedIdsUseExplicitEmptySessionWithoutReopen) {
  start();
  ASSERT_FALSE(runtime->open_session_async("s").get());
  auto event = output();
  GenerationOptions options;
  options.max_new_tokens = 2;
  options.seed = 42;
  options.sampling.temperature = 0.7f;
  auto handle = submit(
      event,
      PromptInput{{make_text_input("hello"), make_token_input({42, 0})}},
      options);
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->text, "AB");
  EXPECT_EQ(event->terminals, 1u);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(event->terminal->stats.prompt_tokens, 4u);
  EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 4u);
  EXPECT_EQ(event->terminal->stats.completion_tokens, 2u);
  EXPECT_EQ(event->terminal->stats.session_reset_reason, "new");
  EXPECT_EQ(
      event->terminal->stats.generated_token_ids,
      (std::vector<Token>{100, 101}));
  EXPECT_EQ(executor.opened().size(), 1u);
  ASSERT_FALSE(executor.fed.empty());
  EXPECT_EQ(executor.fed.front(), (std::vector<Token>{10, 11, 42, 0}));
  EXPECT_NE(tokenizer.encoder_thread, std::this_thread::get_id());
  EXPECT_NE(tokenizer.encoder_thread, executor.engine_thread);
  EXPECT_NE(event->sink_thread, tokenizer.encoder_thread);
  EXPECT_NE(event->sink_thread, executor.engine_thread);
  auto sampling = executor.sampling_params(executor.opened().front());
  ASSERT_TRUE(sampling);
  EXPECT_FLOAT_EQ(sampling->temperature, 0.7f);
  EXPECT_EQ(executor.sampling_seed(executor.opened().front()), 42u);
}

TEST_F(
    TextGenerationTest,
    StrictFullPromptContinuationForwardsPendingExactlyOnce) {
  start();
  auto first = output();
  submit(first, ids({10, 11})).wait();
  auto second = output();
  submit(second, ids({10, 11, 100, 12, 13})).wait();
  ASSERT_TRUE(second->terminal);
  EXPECT_EQ(second->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(second->terminal->stats.prompt_tokens, 5u);
  EXPECT_EQ(second->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(second->terminal->stats.prefilled_prompt_tokens, 3u);
  EXPECT_EQ(executor.opened().size(), 1u);
  ASSERT_EQ(executor.fed.size(), 2u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{100, 12, 13}));
  EXPECT_EQ(executor.seen().back().effective_position(), 2);
  EXPECT_EQ(tokenizer.encode_calls.load(), 0);
}

TEST_F(
    TextGenerationTest,
    PublicTerminalCanSubmitStrictContinuationBeforeReturning) {
  watch_callbacks();
  config.max_requests = 1;
  executor.second_step.hold();
  start();
  auto first = output();
  auto continued = output();
  first->blocked.hold();
  auto submitted = std::make_shared<std::promise<GenerateResult>>();
  auto followup = submitted->get_future();
  auto capture = std::make_shared<int>(0);
  std::weak_ptr<int> retained = capture;
  GenerationOptions options;
  options.max_new_tokens = 1;
  options.stop_strings = {"END"};
  auto result = runtime->generate(
      "s",
      ids({10, 11}),
      options,
      [this, first, continued, submitted, capture, options](
          GenerationEvent update) {
        (void)capture;
        const bool terminal = std::holds_alternative<TerminalEvent>(update);
        first->accept(std::move(update));
        if (terminal) {
          // Precreated captures avoid mutating the fixture's events vector
          // here.
          submitted->set_value(runtime->generate(
              "s",
              ids({10, 11, 100, 12, 13}),
              options,
              [continued](GenerationEvent next) {
                continued->accept(std::move(next));
              }));
          first->blocked.enter();
        }
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto handle = std::get<RequestHandle>(result);
  capture.reset();
  ASSERT_TRUE(first->blocked.wait());
  ASSERT_EQ(followup.wait_for(5s), std::future_status::ready);
  auto next_result = followup.get();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(next_result));
  const auto next = std::get<RequestHandle>(next_result);
  ASSERT_TRUE(executor.second_step.wait());
  EXPECT_NE(handle.id(), next.id());
  EXPECT_FALSE(handle.done());
  EXPECT_FALSE(next.done());
  EXPECT_FALSE(handle.error());
  EXPECT_FALSE(retained.expired());
  // The gate follows recording the terminal, and no later callback writes
  // first.
  ASSERT_TRUE(first->terminal);
  const auto delivered = *first->terminal;
  EXPECT_EQ(first->text, "A");
  EXPECT_EQ(first->terminals, 1u);
  EXPECT_EQ(delivered.finish_reason, FinishReason::Length);
  EXPECT_FALSE(delivered.error);
  EXPECT_EQ(delivered.stats.prompt_tokens, 2u);
  EXPECT_EQ(delivered.stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(delivered.stats.prefilled_prompt_tokens, 2u);
  EXPECT_EQ(delivered.stats.completion_tokens, 1u);
  EXPECT_EQ(delivered.stats.session_reset_reason, "new");
  EXPECT_EQ(delivered.stats.generated_token_ids, (std::vector<Token>{100}));
  EXPECT_GE(delivered.stats.prefill_ms, 0.0);
  EXPECT_GE(delivered.stats.decode_ms, 0.0);
  EXPECT_GE(delivered.stats.total_ms, delivered.stats.prefill_ms);
  handle.cancel();
  first->blocked.release();
  executor.second_step.release();
  ASSERT_TRUE(wait_until([&] { return handle.done() && next.done(); }));
  handle.wait(); // Already done: also exercise the public lifetime barrier.
  EXPECT_TRUE(retained.expired());
  EXPECT_FALSE(handle.error());
  EXPECT_FALSE(next.error());
  expect_same_terminal(*first->terminal, delivered);
  ASSERT_TRUE(continued->terminal);
  EXPECT_EQ(continued->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(continued->terminal->stats.prompt_tokens, 5u);
  EXPECT_EQ(continued->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(continued->terminal->stats.prefilled_prompt_tokens, 3u);
  EXPECT_EQ(
      continued->terminal->stats.generated_token_ids,
      (std::vector<Token>{101}));
  EXPECT_EQ(executor.opened().size(), 1u);
  ASSERT_EQ(executor.fed.size(), 2u);
  EXPECT_EQ(executor.fed.front(), (std::vector<Token>{10, 11}));
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{100, 12, 13}));
  EXPECT_EQ(executor.seen().back().effective_position(), 2);
}

TEST_F(TextGenerationTest, EqualShorterAndMismatchedHistoriesColdReplay) {
  start();
  submit(output(), ids({10, 11})).wait();
  for (const auto& item :
       std::vector<std::pair<std::vector<Token>, std::string>>{
           {{10, 11, 100}, "equal"},
           {{10}, "mismatch"},
           {{20, 21}, "mismatch"}}) {
    auto event = output();
    submit(event, ids(item.first)).wait();
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminal->stats.session_reset_reason, item.second);
    EXPECT_EQ(event->terminal->stats.reused_prompt_tokens, 0u);
    EXPECT_EQ(
        event->terminal->stats.prefilled_prompt_tokens, item.first.size());
  }
  EXPECT_EQ(executor.opened().size(), 4u);
}

TEST_F(TextGenerationTest, PreparationAndOptionFailurePreserveExistingHistory) {
  start();
  submit(output(), ids({10, 11})).wait();
  auto bad_prompt = output();
  auto failed = submit(bad_prompt, PromptInput{{make_text_input("missing")}});
  failed.wait();
  ASSERT_TRUE(failed.error());
  EXPECT_EQ(failed.error()->code, ErrorCode::InvalidArgument);
  for (int i = 0; i < 5; ++i) {
    GenerationOptions bad;
    if (i == 0) {
      bad.max_new_tokens = 0;
    }
    if (i == 1) {
      bad.sampling.temperature = -1;
    }
    if (i == 2) {
      bad.sampling.top_p = 2;
    }
    if (i == 3) {
      bad.sampling.top_k = -1;
    }
    if (i == 4) {
      bad.stop_strings = {""};
    }
    auto event = output();
    auto handle = submit(event, ids({42}), bad);
    handle.wait();
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminal->finish_reason, FinishReason::Failed);
    EXPECT_EQ(event->terminal->error->code, ErrorCode::InvalidArgument);
  }
  auto empty = output();
  submit(empty, {}).wait();
  ASSERT_TRUE(empty->terminal);
  EXPECT_EQ(empty->terminal->error->code, ErrorCode::InvalidArgument);
  auto continued = output();
  submit(continued, ids({10, 11, 100, 12})).wait();
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opened().size(), 1u);
}

TEST_F(TextGenerationTest, UnsupportedModalitiesPreserveExistingHistory) {
  start();
  submit(output(), ids({10, 11})).wait();
  const auto steps = executor.steps.load();
  for (const auto& segment : {
           make_image_input(Image{}),
           make_audio_input(Audio{}),
           make_raw_audio_input(RawAudio{}),
       }) {
    auto event = output();
    auto handle = submit(
        event,
        PromptInput{{make_token_input({42}), segment, make_token_input({43})}});
    handle.wait();
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminals, 1u);
    EXPECT_TRUE(event->text.empty());
    EXPECT_EQ(event->terminal->finish_reason, FinishReason::Failed);
    ASSERT_TRUE(event->terminal->error);
    EXPECT_EQ(event->terminal->error->code, ErrorCode::InvalidArgument);
    ASSERT_TRUE(handle.error());
    EXPECT_EQ(handle.error()->code, ErrorCode::InvalidArgument);
    EXPECT_EQ(executor.steps.load(), steps);
    EXPECT_EQ(executor.opened().size(), 1u);
  }
  auto continued = output();
  submit(continued, ids({10, 11, 100, 12})).wait();
  ASSERT_TRUE(continued->terminal);
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.opened().size(), 1u);
}

TEST_F(TextGenerationTest, ReplayOpenFailureLeavesReservedUnavailableSession) {
  start();
  submit(output(), ids({10, 11})).wait();
  executor.refuse_open = true;
  auto event = output();
  auto handle = submit(event, ids({20, 21}));
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Failed);
  EXPECT_TRUE(handle.error());
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  auto unavailable = runtime->open_session_async("s").get();
  ASSERT_TRUE(unavailable);
  EXPECT_EQ(unavailable->code, ErrorCode::NotReady);
  executor.refuse_open = false;
  EXPECT_FALSE(runtime->reset_session_async("s").get());
  auto fresh = output();
  submit(fresh, ids({20, 21})).wait();
  EXPECT_EQ(fresh->terminal->stats.session_reset_reason, "new");
}

TEST_F(TextGenerationTest, EosIsLogicalHistoryButNotVisibleTextOrReplayIds) {
  config.default_stop_tokens = {9};
  executor.script = {100, 9};
  executor.burst = 2;
  start();
  GenerationOptions options;
  options.max_new_tokens = 10;
  options.stop_tokens = {8};
  auto first = output();
  submit(first, ids({10, 11}), options).wait();
  EXPECT_EQ(first->text, "A");
  ASSERT_TRUE(first->terminal);
  EXPECT_EQ(first->terminal->finish_reason, FinishReason::Stop);
  EXPECT_EQ(first->terminal->stats.completion_tokens, 1u);
  EXPECT_EQ(
      first->terminal->stats.generated_token_ids, (std::vector<Token>{100}));
  auto second = output();
  submit(second, ids({10, 11, 100, 9, 12})).wait();
  EXPECT_EQ(second->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(second->terminal->stats.reused_prompt_tokens, 3u);
  EXPECT_EQ(second->terminal->stats.prefilled_prompt_tokens, 2u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{9, 12}));
}

TEST_F(
    TextGenerationTest,
    StringStopInsideSpeculativeTerminalPrecedesStatsAndDirtyReplay) {
  tokenizer.pieces = {{100, "hello EN"}, {101, "Dhidden"}, {102, "ignored"}};
  executor.burst = 3;
  start();
  GenerationOptions options;
  options.max_new_tokens = 3;
  options.stop_strings = {"END"};
  auto event = output();
  submit(event, ids({10, 11}), options).wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->text, "hello ");
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Stop);
  EXPECT_EQ(event->terminal->stats.completion_tokens, 2u);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  auto replay = output();
  submit(replay, ids({10, 11, 100, 101, 102, 12}), options).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
  EXPECT_EQ(executor.opened().size(), 2u);
}

TEST_F(TextGenerationTest, Utf8AndStopLookbehindFlushBeforeTerminal) {
  tokenizer.pieces = {{100, "a\xE4"}, {101, "\xB8"}, {102, "\x96"}};
  start();
  GenerationOptions options;
  options.max_new_tokens = 3;
  options.stop_strings = {"END"};
  auto event = output();
  submit(event, ids({10, 11}), options).wait();
  EXPECT_EQ(event->text, "a\xE4\xB8\x96");
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(
      event->terminal->stats.generated_token_ids,
      (std::vector<Token>{100, 101, 102}));
  EXPECT_EQ(event->terminals, 1u);
}

TEST_F(TextGenerationTest, DecodeFailureCancelsAndInvalidatesHistory) {
  tokenizer.pieces.erase(101);
  executor.burst = 3;
  start();
  GenerationOptions options;
  options.max_new_tokens = 3;
  auto event = output();
  auto handle = submit(event, ids({10, 11}), options);
  handle.wait();
  EXPECT_EQ(event->text, "A");
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Failed);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  auto next = output();
  submit(next, ids({10, 11, 100, 101, 102, 12})).wait();
  EXPECT_EQ(next->terminal->stats.session_reset_reason, "dirty");
}

TEST_F(
    TextGenerationTest,
    AdmissionReturnsBeforeEncodingAndRejectedRequestsHaveNoCallback) {
  config.max_requests = 1;
  tokenizer.encoding.hold();
  start();
  auto event = output();
  auto handle = submit(event, PromptInput{{make_text_input("hello")}});
  ASSERT_NE(handle.id(), 0u);
  ASSERT_TRUE(tokenizer.encoding.wait());
  EXPECT_EQ(runtime->info().max_sessions, 4u);
  std::atomic<int> callbacks{0};
  auto rejected = runtime->generate(
      "other", ids({10, 11}), {}, [&](GenerationEvent) { ++callbacks; });
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  handle.cancel();
  tokenizer.encoding.release();
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  EXPECT_TRUE(executor.opened().empty());
  EXPECT_EQ(callbacks.load(), 0);
}

TEST_F(TextGenerationTest, SmallPublicSinkDestructionCanReenterInfo) {
  watch_callbacks();
  config.max_requests = 1;
  tokenizer.encoding.hold();
  start();
  // Exercise libc++ inline callable moves as well as final capture cleanup.
  struct SmallSink {
    ServingRuntime* runtime;
    SmallSinkState* state;

    ~SmallSink() {
      (void)runtime->info();
      ++state->destructions;
    }

    void operator()(GenerationEvent) const {
      ++state->calls;
    }
  };
  static_assert(sizeof(SmallSink) == 2 * sizeof(void*));
  static_assert(std::is_nothrow_copy_constructible<SmallSink>::value);
  GenerationOptions options;
  options.max_new_tokens = 1;
  auto result = runtime->generate(
      "s",
      PromptInput{{make_text_input("hello")}},
      options,
      SmallSink{runtime.get(), &small_sink_state});
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto handle = std::get<RequestHandle>(result);
  ASSERT_TRUE(tokenizer.encoding.wait());
  const auto before_rejection = small_sink_state.destructions.load();
  auto rejected = runtime->generate(
      "other",
      ids({20, 21}),
      options,
      SmallSink{runtime.get(), &small_sink_state});
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  EXPECT_GT(small_sink_state.destructions.load(), before_rejection);
  EXPECT_EQ(small_sink_state.calls.load(), 0);
  const auto before_completion = small_sink_state.destructions.load();
  tokenizer.encoding.release();
  ASSERT_TRUE(wait_until([&] { return handle.done(); }));
  EXPECT_FALSE(handle.error());
  EXPECT_GT(small_sink_state.destructions.load(), before_completion);
  EXPECT_EQ(small_sink_state.calls.load(), 2); // Rendered text and terminal.
}

TEST_F(TextGenerationTest, QueuedCancellationSkipsPreparationAndSessionOpen) {
  tokenizer.encoding.hold();
  start();
  auto first = submit(output(), PromptInput{{make_text_input("hello")}});
  ASSERT_TRUE(tokenizer.encoding.wait());
  auto event = output();
  auto cancelled =
      submit(event, PromptInput{{make_text_input("hello")}}, {}, "other");
  cancelled.cancel();
  tokenizer.encoding.release();
  first.wait();
  cancelled.wait();
  EXPECT_EQ(tokenizer.encode_calls.load(), 1);
  EXPECT_EQ(executor.opened().size(), 1u);
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 0u);
}

TEST_F(TextGenerationTest, CancelledPartialPrefillIsNotFullPromptResidency) {
  executor.second_step.hold();
  start(2, 3, 1);
  auto event = output();
  auto handle = submit(event, ids({10, 11, 12, 13, 14, 15}));
  ASSERT_TRUE(executor.second_step.wait());
  handle.cancel();
  executor.second_step.release();
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(event->terminal->stats.prompt_tokens, 6u);
  EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 4u);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  auto replay = output();
  submit(replay, ids({10, 11, 12, 13, 14, 15})).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
  EXPECT_EQ(replay->terminal->stats.prefilled_prompt_tokens, 6u);
  EXPECT_EQ(executor.opened().size(), 2u);
}

TEST_F(TextGenerationTest, CloseDuringPrefillPreservesPhysicalWorkStatistics) {
  watch_callbacks();
  executor.second_step.hold();
  start(2, 3, 1);
  auto event = output();
  auto handle = submit(event, ids({10, 11, 12, 13, 14, 15}));
  ASSERT_TRUE(executor.second_step.wait());
  auto closed = runtime->close_session_async("s");
  // Logical retirement/cancellation precedes the delivery-lifetime ACK.
  ASSERT_TRUE(wait_until([&] { return runtime->info().active_sessions == 0; }));
  auto control = runtime->close_session_async("absent");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  EXPECT_EQ(closed.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(handle.done());
  executor.second_step.release();
  ASSERT_EQ(closed.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(closed.get());
  ASSERT_TRUE(handle.done());
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(event->terminal->stats.prompt_tokens, 6u);
  EXPECT_EQ(event->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 4u);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
}

TEST_F(TextGenerationTest, TextSinkHoldsNamedOwnershipUntilCommit) {
  start();
  auto slow = output();
  slow->block_first = true;
  slow->blocked.hold();
  auto first = submit(slow, ids({10, 11}));
  ASSERT_TRUE(slow->blocked.wait());
  auto other = output();
  auto other_handle =
      submit(other, PromptInput{{make_text_input("hello")}}, {}, "other");
  EXPECT_FALSE(first.done());
  auto busy = output();
  auto busy_handle = submit(busy, ids({10, 11, 100, 12}));
  // Fence the control-side Busy decision before allowing the first commit.
  ASSERT_FALSE(runtime->open_session_async("s").get());
  slow->blocked.release();
  first.wait();
  other_handle.wait();
  busy_handle.wait();
  EXPECT_EQ(other->text, "A");
  ASSERT_TRUE(busy->terminal);
  ASSERT_TRUE(busy->terminal->error);
  EXPECT_EQ(busy->terminal->error->code, ErrorCode::SessionBusy);
  auto continued = output();
  submit(continued, ids({10, 11, 100, 12})).wait();
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
}

class TextGenerationFenceTest : public TextGenerationTest,
                                public ::testing::WithParamInterface<bool> {
 protected:
  std::future<LifecycleResult> transition() {
    return GetParam() ? runtime->reset_session_async("s")
                      : runtime->close_session_async("s");
  }
};

TEST_P(TextGenerationFenceTest, AckIncludesSubmittingThreadInlineSinkOwners) {
#if !defined(_LIBCPP_VERSION) || defined(_LIBCPP_ABI_OPTIMIZED_FUNCTION)
  GTEST_SKIP() << "Requires libc++ inline std::function cloning on move";
#else
  struct InlineSink {
    InlineSinkState* state;
    int depth = 0;

    explicit InlineSink(InlineSinkState* state) noexcept : state(state) {
      ++state->live_owners;
    }
    InlineSink(const InlineSink& other) noexcept
        : state(other.state), depth(other.depth + 1) {
      ++state->live_owners;
      ++state->copies;
    }
    InlineSink(InlineSink&& other) noexcept
        : state(other.state), depth(other.depth) {
      other.state = nullptr;
    }
    ~InlineSink() {
      if (!state) {
        return;
      }
      if (depth == 0) {
        ++state->original_destructions;
      } else {
        ++state->copied_destructions;
        if (state->cleanup &&
            std::this_thread::get_id() == state->submitting_thread) {
          state->cleanup_entered.store(true);
          state->cleanup->enter();
        }
      }
      --state->live_owners;
    }
    void operator()(GenerationEvent event) const {
      if (std::holds_alternative<TerminalEvent>(event)) {
        ++state->terminals;
      }
    }
  };
  static_assert(sizeof(InlineSink) <= 2 * sizeof(void*));
  static_assert(std::is_nothrow_copy_constructible<InlineSink>::value);
  using Sink = std::function<void(GenerationEvent)>;
  auto& state = inline_sink_state;
  bool supported = true;
  {
    Sink incoming{InlineSink{&state}};
    supported &= state.live_owners.load() == 1 && state.copies.load() == 0;
    Sink local;
    local = std::move(incoming);
    supported &= incoming && local && state.live_owners.load() == 2 &&
        state.copies.load() == 1;
    std::optional<Sink> parameter(std::move(local));
    auto published = std::make_shared<Sink>(std::move(*parameter));
    // Probe the old move-assignment/optional/shared-object path. There must
    // be no positive-depth destructor before publication, or gating could
    // stop admission rather than expose a returning caller's retained owner.
    supported &= local && *parameter && *published &&
        state.live_owners.load() == 4 && state.copies.load() == 3 &&
        state.original_destructions.load() == 0 &&
        state.copied_destructions.load() == 0;
  }
  supported &= state.live_owners.load() == 0 &&
      state.original_destructions.load() == 1 &&
      state.copied_destructions.load() == 3;
  state.copies = 0;
  state.original_destructions = 0;
  state.copied_destructions = 0;
  {
    Sink incoming{InlineSink{&state}};
    Sink target;
    target.swap(incoming);
    // The fixed path empties incoming before publication. Only the original
    // depth-zero owner is destroyed here, which the regression never gates.
    supported &= !incoming && target && state.live_owners.load() == 1 &&
        state.copies.load() == 1 && state.original_destructions.load() == 1 &&
        state.copied_destructions.load() == 0;
  }
  supported &=
      state.live_owners.load() == 0 && state.copied_destructions.load() == 1;
  if (!supported) {
    GTEST_SKIP() << "Unsupported std::function inline move/swap behavior";
  }

  watch_callbacks();
  start();
  cleanup_checkpoint.hold();
  state.cleanup = &cleanup_checkpoint;
  auto submitted = std::make_shared<std::promise<GenerateResult>>();
  auto returned = submitted->get_future();
  submitting_caller = std::thread([this, submitted] {
    inline_sink_state.submitting_thread = std::this_thread::get_id();
    GenerationOptions options;
    options.max_new_tokens = 1;
    // A prvalue function avoids a caller-owned source. Complete this full
    // expression before reporting return, including parameter destruction.
    auto result = runtime->generate(
        "s", ids({10, 11}), options, Sink{InlineSink{&inline_sink_state}});
    submitted->set_value(std::move(result));
  });
  ASSERT_TRUE(wait_until([&] {
    return returned.wait_for(0s) == std::future_status::ready ||
        state.cleanup_entered.load();
  }));
  // A copied-source destructor in the old implementation is now held before
  // decrementing live_owners. Dispatcher destruction remains unblocked.
  auto ack = transition();
  ASSERT_EQ(ack.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_EQ(state.terminals.load(), 1);
  EXPECT_EQ(state.live_owners.load(), 0)
      << "ACK preceded destruction of an inline sink owner on the caller";

  cleanup_checkpoint.release();
  ASSERT_EQ(returned.wait_for(5s), std::future_status::ready);
  auto result = returned.get();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  EXPECT_TRUE(std::get<RequestHandle>(result).done());
  submitting_caller.join();
#endif
}

TEST_P(
    TextGenerationFenceTest,
    RenderedTextFlushTerminalAndCaptureCleanupPrecedeColdFollowup) {
  watch_callbacks();
  tokenizer.pieces[100] = "AEN";
  tokenizer.encoding.hold();
  start();
  auto slow = output();
  slow->block_first = true;
  slow->blocked.hold();
  slow->block_terminal = true;
  slow->terminal_blocked.hold();
  delivery_checkpoint.hold();
  slow->on_text = [this, count = 0]() mutable {
    if (++count == 2) {
      delivery_checkpoint.enter();
    }
  };
  auto cleaned = std::make_shared<std::atomic<bool>>(false);
  auto capture = std::shared_ptr<int>(new int(0), [this, cleaned](int* value) {
    cleanup_checkpoint.enter();
    delete value;
    cleaned->store(true);
  });
  GenerationOptions options;
  options.max_new_tokens = 1;
  options.stop_strings = {"END"};
  auto result = runtime->generate(
      "s", ids({10, 11}), options, [slow, capture](GenerationEvent update) {
        (void)capture;
        slow->accept(std::move(update));
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto old = std::get<RequestHandle>(result);
  ASSERT_TRUE(slow->blocked.wait());
  cleanup_checkpoint.hold();
  capture.reset();

  auto ack = transition();
  auto replacement = output();
  // This would be an exact-prefix continuation without the cold transition.
  const auto next = submit(
      replacement,
      PromptInput{{make_text_input("hello"), make_token_input({100, 12})}});
  auto other_open = runtime->open_session_async("other");
  ASSERT_EQ(other_open.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(other_open.get());
  auto other = output();
  const auto other_handle = submit(other, ids({20, 21}), {}, "other");
  ASSERT_TRUE(wait_until([&] { return executor.seen().size() == 2; }));
  // Control and execution progress, but the shared sink cannot deliver other.
  EXPECT_FALSE(other_handle.done());
  EXPECT_EQ(runtime->info().active_sessions, GetParam() ? 2u : 1u);
  EXPECT_EQ(executor.opened().size(), GetParam() ? 3u : 2u);
  const auto expect_pending = [&] {
    EXPECT_EQ(ack.wait_for(0s), std::future_status::timeout);
    EXPECT_FALSE(old.done());
    EXPECT_FALSE(next.done());
    EXPECT_EQ(tokenizer.encode_calls.load(), 0);
    EXPECT_FALSE(cleaned->load());
  };
  expect_pending();

  slow->blocked.release();
  ASSERT_TRUE(delivery_checkpoint.wait());
  auto control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  // "A" was rendered; the stop lookbehind "EN" is in the final flush sink.
  EXPECT_EQ(slow->text, "A");
  EXPECT_EQ(slow->terminals, 0u);
  expect_pending();
  delivery_checkpoint.release();
  ASSERT_TRUE(slow->terminal_blocked.wait());
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  ASSERT_TRUE(slow->terminal);
  EXPECT_EQ(slow->text, "AEN");
  EXPECT_EQ(slow->terminals, 1u);
  const auto delivered = *slow->terminal;
  // Engine Length was already published before close/reset was accepted.
  EXPECT_EQ(delivered.finish_reason, FinishReason::Length);
  EXPECT_EQ(delivered.stats.completion_tokens, 1u);
  EXPECT_EQ(delivered.stats.generated_token_ids, (std::vector<Token>{100}));
  expect_pending();
  slow->terminal_blocked.release();
  ASSERT_TRUE(cleanup_checkpoint.wait());
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  expect_pending();

  cleanup_checkpoint.release();
  ASSERT_EQ(ack.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_TRUE(old.done());
  EXPECT_TRUE(cleaned->load());
  expect_same_terminal(*slow->terminal, delivered);
  // A later generation is outside the cutoff: its held preparation cannot
  // prevent ACK, and preparation must not start on control before ACK.
  ASSERT_TRUE(tokenizer.encoding.wait());
  EXPECT_FALSE(next.done());
  tokenizer.encoding.release();
  ASSERT_TRUE(wait_until([&] { return next.done() && other_handle.done(); }));
  EXPECT_FALSE(next.error());
  EXPECT_FALSE(other_handle.error());
  EXPECT_EQ(other->text, "AEN");
  ASSERT_TRUE(replacement->terminal);
  EXPECT_EQ(replacement->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(replacement->terminal->stats.session_reset_reason, "new");
  EXPECT_EQ(replacement->terminal->stats.prompt_tokens, 4u);
  EXPECT_EQ(replacement->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(replacement->terminal->stats.prefilled_prompt_tokens, 4u);
  EXPECT_EQ(
      replacement->terminal->stats.generated_token_ids,
      (std::vector<Token>{100}));
  EXPECT_EQ(executor.opened().size(), 3u);
  ASSERT_EQ(executor.fed.size(), 3u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{10, 11, 100, 12}));

  auto continued = output();
  const auto continuation = submit(continued, ids({10, 11, 100, 12, 100, 13}));
  ASSERT_TRUE(wait_until([&] { return continuation.done(); }));
  ASSERT_TRUE(continued->terminal);
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(continued->terminal->stats.reused_prompt_tokens, 4u);
  EXPECT_EQ(continued->terminal->stats.prefilled_prompt_tokens, 2u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{100, 13}));
  // Neither deferred fence polling nor old text finalization reopens again.
  EXPECT_EQ(executor.opened().size(), 3u);
}

TEST_P(
    TextGenerationFenceTest,
    RetiringPublicTerminalCanEnqueueTransitionAndNextGeneration) {
  watch_callbacks();
  config.max_requests = 1;
  tokenizer.encoding.hold();
  start();
  auto first = output();
  auto replacement = output();
  first->blocked.hold();
  auto acknowledged = std::make_shared<std::promise<LifecycleResult>>();
  auto ack = acknowledged->get_future();
  auto submitted = std::make_shared<std::promise<GenerateResult>>();
  auto followup = submitted->get_future();
  GenerationOptions options;
  options.max_new_tokens = 1;
  options.stop_strings = {"END"};
  auto result = runtime->generate(
      "s",
      ids({10, 11}),
      options,
      [service = runtime.get(),
       first,
       replacement,
       acknowledged,
       submitted,
       reset = GetParam()](GenerationEvent update) {
        const bool terminal = std::holds_alternative<TerminalEvent>(update);
        first->accept(std::move(update));
        if (!terminal) {
          return;
        }
        LifecycleCallback on_ack = [acknowledged](LifecycleResult error) {
          acknowledged->set_value(std::move(error));
        };
        if (reset) {
          service->reset_session_async("s", std::move(on_ack));
        } else {
          service->close_session_async("s", std::move(on_ack));
        }
        GenerationOptions next_options;
        next_options.max_new_tokens = 1;
        submitted->set_value(service->generate(
            "s",
            PromptInput{
                {make_text_input("hello"), make_token_input({100, 12})}},
            next_options,
            [replacement](GenerationEvent next) {
              replacement->accept(std::move(next));
            }));
        first->blocked.enter();
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto old = std::get<RequestHandle>(result);
  ASSERT_TRUE(first->blocked.wait());
  ASSERT_EQ(followup.wait_for(5s), std::future_status::ready);
  auto next_result = followup.get();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(next_result));
  const auto next = std::get<RequestHandle>(next_result);
  EXPECT_NE(old.id(), next.id());
  // The old request no longer consumes admission, but must still fence ACK.
  auto control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  EXPECT_EQ(ack.wait_for(0s), std::future_status::timeout);
  EXPECT_EQ(tokenizer.encode_calls.load(), 0);
  EXPECT_FALSE(old.done());
  EXPECT_FALSE(next.done());
  ASSERT_TRUE(first->terminal);
  const auto delivered = *first->terminal;
  EXPECT_EQ(first->text, "A");
  EXPECT_EQ(delivered.finish_reason, FinishReason::Length);
  first->blocked.release();
  ASSERT_EQ(ack.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_TRUE(old.done());
  ASSERT_TRUE(tokenizer.encoding.wait());
  EXPECT_FALSE(next.done());
  tokenizer.encoding.release();
  ASSERT_TRUE(wait_until([&] { return next.done(); }));
  EXPECT_FALSE(old.error());
  EXPECT_FALSE(next.error());
  EXPECT_EQ(first->terminals, 1u);
  expect_same_terminal(*first->terminal, delivered);
  ASSERT_TRUE(replacement->terminal);
  EXPECT_EQ(replacement->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(replacement->terminal->stats.session_reset_reason, "new");
  EXPECT_EQ(replacement->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(replacement->terminal->stats.prefilled_prompt_tokens, 4u);
  EXPECT_EQ(executor.opened().size(), 3u);
  ASSERT_EQ(executor.fed.size(), 2u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{10, 11, 100, 12}));
}

TEST_P(TextGenerationFenceTest, AckWaitsForEveryPriorPublicGeneration) {
  watch_callbacks();
  start();
  auto first = output();
  first->block_terminal = true;
  first->terminal_blocked.hold();
  const auto old = submit(first, ids({10, 11}));
  ASSERT_TRUE(first->terminal_blocked.wait());
  auto second = output();
  second->block_terminal = true;
  second->terminal_blocked.hold();
  const auto prior = submit(second, ids({10, 11, 100, 12}));
  ASSERT_TRUE(wait_until([&] { return executor.seen().size() == 2; }));
  auto ack = transition();
  auto control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  EXPECT_EQ(ack.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(old.done());
  EXPECT_FALSE(prior.done());

  first->terminal_blocked.release();
  ASSERT_TRUE(second->terminal_blocked.wait());
  ASSERT_TRUE(wait_until([&] { return old.done(); }));
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(control.get());
  EXPECT_EQ(ack.wait_for(0s), std::future_status::timeout);
  EXPECT_FALSE(prior.done());
  ASSERT_TRUE(second->terminal);
  EXPECT_EQ(second->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(second->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(second->terminal->stats.prefilled_prompt_tokens, 2u);
  second->terminal_blocked.release();
  ASSERT_EQ(ack.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_TRUE(old.done());
  EXPECT_TRUE(prior.done());
  EXPECT_EQ(first->terminals, 1u);
  EXPECT_EQ(second->terminals, 1u);
  EXPECT_EQ(runtime->info().active_sessions, GetParam() ? 2u : 1u);
  EXPECT_EQ(executor.opened().size(), GetParam() ? 3u : 2u);
}

INSTANTIATE_TEST_SUITE_P(
    CloseAndReset,
    TextGenerationFenceTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Reset" : "Close";
    });

TEST_F(TextGenerationTest, UnsetAndExplicitBudgetsUseRemainingContext) {
  config.max_context_length = 5;
  config.default_max_new_tokens = 1;
  start();
  for (auto limit :
       {std::optional<std::int32_t>{}, std::optional<std::int32_t>{99}}) {
    auto event = output();
    GenerationOptions options;
    options.max_new_tokens = limit;
    auto result = runtime->generate(
        std::nullopt, ids({10, 11}), options, [event](GenerationEvent update) {
          event->accept(std::move(update));
        });
    ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
    std::get<RequestHandle>(result).wait();
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminal->finish_reason, FinishReason::Length);
    EXPECT_EQ(event->terminal->stats.completion_tokens, 3u);
    EXPECT_EQ(runtime->info().active_sessions, 0u);
  }
  auto full = output();
  submit(full, ids({1, 2, 3, 4, 5})).wait();
  ASSERT_TRUE(full->terminal);
  EXPECT_EQ(full->terminal->error->code, ErrorCode::InvalidArgument);
}

TEST_F(TextGenerationTest, UnknownContextUsesServiceDefaultOnlyForUnsetLimit) {
  config.max_context_length = 0;
  config.default_max_new_tokens = 2;
  start();
  auto event = output();
  auto result = runtime->generate(
      std::nullopt, ids({10, 11}), {}, [event](GenerationEvent update) {
        event->accept(std::move(update));
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  std::get<RequestHandle>(result).wait();
  EXPECT_EQ(event->terminal->stats.completion_tokens, 2u);
  EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 2u);
  EXPECT_GE(event->terminal->stats.total_ms, event->terminal->stats.prefill_ms);
}

TEST_F(TextGenerationTest, StreamingStringStopIgnoresLaterRawTokens) {
  tokenizer.pieces = {{100, "hello EN"}, {101, "Dhidden"}, {102, "ignored"}};
  executor.burst = 3;
  start();
  GenerationOptions options;
  options.max_new_tokens = 8;
  options.stop_strings = {"END"};
  auto event = output();
  submit(event, ids({10, 11}), options).wait();
  EXPECT_EQ(event->text, "hello ");
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Stop);
  EXPECT_EQ(event->terminal->stats.completion_tokens, 2u);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
}

TEST_F(TextGenerationTest, OverflowIsFailureAndForcesColdReplay) {
  config.max_tokens_per_request = 1;
  executor.burst = 3;
  start();
  GenerationOptions options;
  options.max_new_tokens = 3;
  auto event = output();
  auto handle = submit(event, ids({10, 11}), options);
  handle.wait();
  EXPECT_TRUE(event->text.empty());
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Failed);
  EXPECT_EQ(event->terminal->error->code, ErrorCode::CapacityExceeded);
  EXPECT_FALSE(event->terminal->stats.generated_token_ids);
  executor.burst = 1;
  auto replay = output();
  submit(replay, ids({10, 11, 100, 101, 102, 12})).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
}

TEST_F(TextGenerationTest, LifecycleOnlyConstructorRejectsTextWithoutCallback) {
  ServingRuntime lifecycle(
      executor, batching::DecodeFirstScheduler::create(32, 4, 8), config);
  bool callback = false;
  auto result = lifecycle.generate(
      "s", ids({10, 11}), {}, [&](GenerationEvent) { callback = true; });
  ASSERT_TRUE(std::holds_alternative<ServingError>(result));
  EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::NotReady);
  lifecycle.shutdown();
  EXPECT_FALSE(callback);
}

#if ET_HAS_EXCEPTIONS
TEST_F(
    TextGenerationTest,
    ThrowingPublicTerminalPreservesLengthAndStopHistory) {
  watch_callbacks();
  config.max_requests = 1;
  executor.burst = 2;
  for (const auto reason : {FinishReason::Length, FinishReason::Stop}) {
    SCOPED_TRACE(reason == FinishReason::Length ? "Length" : "Stop");
    const auto fed_before = executor.fed.size();
    const auto opened_before = executor.opened().size();
    start();
    auto event = output();
    event->blocked.hold();
    GenerationOptions options;
    options.max_new_tokens = reason == FinishReason::Length ? 2 : 8;
    if (reason == FinishReason::Stop) {
      options.stop_tokens = {101};
    }
    auto result = runtime->generate(
        "s", ids({10, 11}), options, [event](GenerationEvent update) {
          const bool terminal = std::holds_alternative<TerminalEvent>(update);
          event->accept(std::move(update));
          if (terminal) {
            event->blocked.enter();
            throw std::runtime_error("terminal transport");
          }
        });
    ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
    const auto handle = std::get<RequestHandle>(result);
    ASSERT_TRUE(event->blocked.wait());
    ASSERT_TRUE(event->terminal);
    const auto delivered = *event->terminal;
    const auto expected_ids = reason == FinishReason::Length
        ? std::vector<Token>{100, 101}
        : std::vector<Token>{100};
    EXPECT_EQ(delivered.finish_reason, reason);
    EXPECT_FALSE(delivered.error);
    EXPECT_EQ(delivered.stats.generated_token_ids, expected_ids);
    EXPECT_EQ(delivered.stats.completion_tokens, expected_ids.size());
    EXPECT_EQ(delivered.stats.prompt_tokens, 2u);
    EXPECT_EQ(delivered.stats.reused_prompt_tokens, 0u);
    EXPECT_EQ(delivered.stats.prefilled_prompt_tokens, 2u);
    EXPECT_EQ(delivered.stats.session_reset_reason, "new");
    EXPECT_EQ(event->text, reason == FinishReason::Length ? "AB" : "A");
    EXPECT_FALSE(handle.done());
    EXPECT_FALSE(handle.error());
    handle.cancel();
    event->blocked.release();
    ASSERT_TRUE(wait_until([&] { return handle.done(); }));
    EXPECT_FALSE(handle.error());
    EXPECT_EQ(event->terminals, 1u);
    expect_same_terminal(*event->terminal, delivered);

    // Both outcomes retain 101 as the pending logical token, even when hidden.
    auto continued = output();
    const auto next = submit(continued, ids({10, 11, 100, 101, 12}));
    ASSERT_TRUE(wait_until([&] { return next.done(); }));
    EXPECT_FALSE(next.error());
    ASSERT_TRUE(continued->terminal);
    EXPECT_EQ(continued->terminal->finish_reason, FinishReason::Length);
    EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
    EXPECT_EQ(continued->terminal->stats.prompt_tokens, 5u);
    EXPECT_EQ(continued->terminal->stats.reused_prompt_tokens, 3u);
    EXPECT_EQ(continued->terminal->stats.prefilled_prompt_tokens, 2u);
    EXPECT_EQ(
        continued->terminal->stats.generated_token_ids,
        (std::vector<Token>{102}));
    EXPECT_EQ(executor.opened().size(), opened_before + 1);
    ASSERT_EQ(executor.fed.size(), fed_before + 2);
    EXPECT_EQ(executor.fed.back(), (std::vector<Token>{101, 12}));
    EXPECT_EQ(executor.seen().back().effective_position(), 3);
    handle.cancel();
    runtime->shutdown();
    EXPECT_TRUE(handle.done());
    EXPECT_FALSE(handle.error());
    expect_same_terminal(*event->terminal, delivered);
  }
}

TEST_F(
    TextGenerationTest,
    ThrowingPublicErrorTerminalPreservesInvalidArgumentAndResidentHistory) {
  watch_callbacks();
  config.max_requests = 1;
  start();
  auto resident = output();
  const auto seeded = submit(resident, ids({10, 11}));
  ASSERT_TRUE(wait_until([&] { return seeded.done(); }));
  ASSERT_FALSE(seeded.error());
  auto event = output();
  event->blocked.hold();
  GenerationOptions options;
  options.max_new_tokens = 1;
  auto result = runtime->generate(
      "s",
      PromptInput{{make_text_input("missing")}},
      options,
      [event](GenerationEvent update) {
        const bool terminal = std::holds_alternative<TerminalEvent>(update);
        event->accept(std::move(update));
        if (terminal) {
          event->blocked.enter();
          throw std::runtime_error("error terminal transport");
        }
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto handle = std::get<RequestHandle>(result);
  ASSERT_TRUE(event->blocked.wait());
  ASSERT_TRUE(event->terminal);
  const auto delivered = *event->terminal;
  EXPECT_EQ(delivered.finish_reason, FinishReason::Failed);
  ASSERT_TRUE(delivered.error);
  EXPECT_EQ(delivered.error->code, ErrorCode::InvalidArgument);
  EXPECT_FALSE(delivered.stats.generated_token_ids);
  EXPECT_EQ(delivered.stats.prompt_tokens, 0u);
  EXPECT_EQ(delivered.stats.completion_tokens, 0u);
  EXPECT_EQ(delivered.stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(delivered.stats.prefilled_prompt_tokens, 0u);
  EXPECT_TRUE(event->text.empty());
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, delivered.error->code);
  EXPECT_EQ(handle.error()->message, delivered.error->message);
  EXPECT_FALSE(handle.done());
  handle.cancel();
  event->blocked.release();
  ASSERT_TRUE(wait_until([&] { return handle.done(); }));
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::InvalidArgument);
  EXPECT_EQ(event->terminals, 1u);
  expect_same_terminal(*event->terminal, delivered);

  auto continued = output();
  const auto next = submit(continued, ids({10, 11, 100, 12}));
  ASSERT_TRUE(wait_until([&] { return next.done(); }));
  EXPECT_FALSE(next.error());
  ASSERT_TRUE(continued->terminal);
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(continued->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(continued->terminal->stats.prefilled_prompt_tokens, 2u);
  EXPECT_EQ(executor.opened().size(), 1u);
  ASSERT_EQ(executor.fed.size(), 2u);
  EXPECT_EQ(executor.fed.back(), (std::vector<Token>{100, 12}));
  handle.cancel();
  runtime->shutdown();
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, delivered.error->code);
  EXPECT_EQ(handle.error()->message, delivered.error->message);
  expect_same_terminal(*event->terminal, delivered);
}

TEST_F(TextGenerationTest, ThrowingPublicSinkDisablesOutputAndDirtiesHistory) {
  start();
  GenerationOptions options;
  options.max_new_tokens = 1;
  auto result =
      runtime->generate("s", ids({10, 11}), options, [](GenerationEvent) {
        throw std::runtime_error("sink");
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  auto handle = std::get<RequestHandle>(result);
  handle.wait();
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  auto replay = output();
  submit(replay, ids({10, 11, 100, 12})).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
}
TEST_F(TextGenerationTest, ThrowingFinalFlushSinkIsNotCalledAgainForTerminal) {
  start();
  GenerationOptions options;
  options.max_new_tokens = 1;
  options.stop_strings = {"END"};
  std::atomic<int> calls{0};
  auto result =
      runtime->generate("s", ids({10, 11}), options, [&](GenerationEvent) {
        ++calls;
        throw std::runtime_error("flush sink");
      });
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  auto handle = std::get<RequestHandle>(result);
  handle.wait();
  EXPECT_EQ(calls.load(), 1);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  auto replay = output();
  submit(replay, ids({10, 11, 100, 12})).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
}
#endif

} // namespace

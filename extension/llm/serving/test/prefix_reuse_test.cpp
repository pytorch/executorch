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
#include <gtest/gtest.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <utility>

using namespace executorch::extension::llm;
using namespace executorch::extension::llm::serving;
using batching::Position;
using batching::SessionId;
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
  bool entered_ = false;
  bool held_ = false;
};

class Tokenizer : public tokenizers::Tokenizer {
 public:
  tokenizers::Error load(const std::string&) override {
    return tokenizers::Error::Ok;
  }
  tokenizers::Result<std::vector<Token>>
  encode(const std::string& text, int8_t, int8_t) const override {
    return std::vector<Token>(text.begin(), text.end());
  }
  tokenizers::Result<std::string> decode(Token, Token token, bool)
      const override {
    return std::string(1, static_cast<char>('A' + token % 26));
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
  struct Clone {
    SessionId source;
    SessionId destination;
    std::vector<Token> prefix;
  };
  struct Feed {
    SessionId session;
    Position position;
    std::vector<Token> tokens;
  };
  Gate executing;
  Gate cloning;
  Gate decoding;
  std::atomic<bool> refuse_clone{false};
  std::atomic<int> clone_calls{0};
  std::atomic<int> peak_rows{0};
  bool reject_after_decode = false;
  std::size_t burst = 1;

  std::optional<SessionId> open_session() override {
    auto session = FakeExecutor::open_session();
    if (session) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      histories_[*session] = {};
      peak_rows.store(std::max(peak_rows.load(), open_count()));
    }
    return session;
  }
  void close_session(SessionId session) override {
    FakeExecutor::close_session(session);
    std::lock_guard<std::mutex> lock(data_mutex_);
    histories_.erase(session);
    produced_.erase(session);
    advanced_.erase(session);
  }
  std::optional<SessionId> clone(SessionId source, Position upto) override {
    ++clone_calls;
    cloning.enter();
    if (refuse_clone.load()) {
      return std::nullopt;
    }
    std::vector<Token> prefix;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      auto it = histories_.find(source);
      if (it == histories_.end() || upto < 0 ||
          static_cast<std::size_t>(upto) > it->second.size() ||
          (reject_after_decode && advanced_[source])) {
        return std::nullopt;
      }
      prefix.assign(it->second.begin(), it->second.begin() + upto);
    }
    auto session = FakeExecutor::open_session();
    if (session) {
      std::lock_guard<std::mutex> lock(data_mutex_);
      histories_[*session] = prefix;
      clones_.push_back(Clone{source, *session, std::move(prefix)});
      peak_rows.store(std::max(peak_rows.load(), open_count()));
    }
    return session;
  }
  bool execute(const batching::BatchInput& batch, batching::BatchOutput& output)
      override {
    executing.enter();
    bool decode = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (const auto& input : batch.inputs) {
        decode = decode || produced_[input.sid] != 0;
      }
    }
    if (decode) {
      decoding.enter();
    }
    if (!FakeExecutor::execute(batch, output)) {
      return false;
    }
    std::lock_guard<std::mutex> lock(data_mutex_);
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      const auto& input = batch.inputs[i];
      const auto begin =
          static_cast<std::size_t>(input.position) + input.offset;
      auto& history = histories_[input.sid];
      EXPECT_LE(begin, history.size());
      history.resize(begin);
      std::vector<Token> fed(
          input.tokens->begin() + input.offset,
          input.tokens->begin() + input.offset + input.size);
      history.insert(history.end(), fed.begin(), fed.end());
      feeds_.push_back(
          Feed{input.sid, static_cast<Position>(begin), std::move(fed)});
      advanced_[input.sid] = advanced_[input.sid] || produced_[input.sid] != 0;
      if (output.outputs[i]) {
        auto& tokens = output.outputs[i]->tokens;
        tokens.clear();
        for (std::size_t j = 0; j < burst; ++j) {
          tokens.push_back(100 + produced_[input.sid]++);
        }
        history.insert(history.end(), tokens.begin(), tokens.end() - 1);
      }
    }
    return true;
  }
  std::vector<Clone> clones() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return clones_;
  }
  std::vector<Feed> feeds() const {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return feeds_;
  }

 private:
  mutable std::mutex data_mutex_;
  std::map<SessionId, std::vector<Token>> histories_;
  std::map<SessionId, std::size_t> produced_;
  std::map<SessionId, bool> advanced_;
  std::vector<Clone> clones_;
  std::vector<Feed> feeds_;
};

struct Events {
  std::optional<TerminalEvent> terminal;
  std::size_t terminals = 0;
  Gate blocked;
  bool block_first = false;
  std::function<void()> on_text;
  void accept(GenerationEvent event) {
    if (std::holds_alternative<TextEvent>(event)) {
      if (block_first) {
        block_first = false;
        blocked.enter();
      }
      if (on_text) {
        on_text();
      }
    } else {
      terminal = std::get<TerminalEvent>(std::move(event));
      ++terminals;
    }
  }
};

class PrefixReuseTest : public ::testing::Test {
 protected:
  Executor executor;
  Tokenizer tokenizer;
  ServingRuntimeConfig config;
  Gate delivery_checkpoint;
  std::unique_ptr<ServingRuntime> runtime;
  std::vector<std::shared_ptr<Events>> events;
  void SetUp() override {
    config.max_sessions = 4;
    config.max_context_length = 64;
    config.prefix_cache_capacity = 2;
  }
  void start(bool provision = true) {
    if (provision) {
      executor.capacity = static_cast<int>(
          config.max_sessions +
          (config.prefix_cache_capacity ? config.prefix_cache_capacity + 1
                                        : 0));
    }
    runtime = std::make_unique<ServingRuntime>(
        executor,
        batching::DecodeFirstScheduler::create(32, 4, 8),
        tokenizer,
        config);
  }
  void TearDown() override {
    delivery_checkpoint.release();
    executor.executing.release();
    executor.cloning.release();
    executor.decoding.release();
    executor.release();
    for (auto& event : events) {
      event->blocked.release();
    }
    runtime.reset();
    EXPECT_EQ(executor.open_count(), 0);
    auto opened = executor.opened();
    auto closed = executor.closed();
    std::sort(opened.begin(), opened.end());
    std::sort(closed.begin(), closed.end());
    EXPECT_EQ(opened, closed);
    EXPECT_LE(
        static_cast<std::size_t>(executor.peak_rows.load()),
        config.max_sessions +
            (config.prefix_cache_capacity ? config.prefix_cache_capacity + 1
                                          : 0));
  }
  std::shared_ptr<Events> output() {
    auto event = std::make_shared<Events>();
    events.push_back(event);
    return event;
  }
  RequestHandle submit(
      const std::shared_ptr<Events>& event,
      std::vector<Token> tokens,
      std::optional<std::string> key = "s",
      int limit = 1,
      float temperature = 0) {
    GenerationOptions options;
    options.max_new_tokens = limit;
    options.sampling.temperature = temperature;
    options.seed = 42;
    auto result = runtime->generate(
        std::move(key),
        PromptInput{{make_token_input(std::move(tokens))}},
        options,
        [event](GenerationEvent update) { event->accept(std::move(update)); });
    if (auto* error = std::get_if<ServingError>(&result)) {
      ADD_FAILURE() << error->message;
      return {};
    }
    return std::get<RequestHandle>(std::move(result));
  }
};

TEST_F(PrefixReuseTest, DisabledByDefaultAndNonGreedyRequestsNeverClone) {
  EXPECT_EQ(ServingRuntimeConfig{}.prefix_cache_capacity, 0u);
  start();
  auto first = output();
  submit(first, {1, 2, 3}, "a", 1, 0.7f).wait();
  EXPECT_EQ(executor.clone_calls.load(), 0);
  submit(output(), {1, 2, 3}, "seed").wait();
  ASSERT_EQ(executor.clone_calls.load(), 1);
  auto second = output();
  submit(second, {1, 2, 3}, "b", 1, 0.7f).wait();
  ASSERT_TRUE(second->terminal);
  EXPECT_EQ(second->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
}

TEST_F(PrefixReuseTest, DisabledCacheDoesNotCaptureOrLookupGreedyRequests) {
  config.prefix_cache_capacity = 0;
  start();
  submit(output(), {1, 2, 3}, "a").wait();
  auto second = output();
  submit(second, {1, 2, 3}, "b").wait();
  ASSERT_TRUE(second->terminal);
  EXPECT_EQ(second->terminal->stats.prefilled_prompt_tokens, 3u);
  EXPECT_EQ(executor.clone_calls.load(), 0);
}

TEST_F(
    PrefixReuseTest,
    NewNamedAndAnonymousSessionsHitWithFreshFinalForwardAndCorrectSid) {
  start();
  auto first = output();
  submit(first, {1, 2, 3}, "seed").wait();
  ASSERT_TRUE(first->terminal);
  EXPECT_EQ(executor.clone_calls.load(), 1);
  EXPECT_EQ(executor.open_count(), 2);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  for (auto key :
       {std::optional<std::string>{"named"}, std::optional<std::string>{}}) {
    auto event = output();
    submit(event, {1, 2, 3}, key).wait();
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminal->finish_reason, FinishReason::Length);
    EXPECT_EQ(event->terminal->stats.prompt_tokens, 3u);
    EXPECT_EQ(event->terminal->stats.reused_prompt_tokens, 2u);
    EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 1u);
    EXPECT_EQ(event->terminal->stats.session_reset_reason, "new");
    EXPECT_EQ(
        event->terminal->stats.generated_token_ids, (std::vector<Token>{100}));
    const auto feeds = executor.feeds();
    EXPECT_EQ(feeds.back().tokens, (std::vector<Token>{3}));
    EXPECT_EQ(feeds.back().position, 2);
    EXPECT_NE(feeds.back().session, feeds.front().session);
    EXPECT_EQ(executor.seen().back().sampling_seed, 42u);
    // The cache never carries the seed generation's pending prediction (100).
    EXPECT_EQ(event->terminal->stats.completion_tokens, 1u);
  }
  EXPECT_EQ(executor.clone_calls.load(), 5); // seed + two lookup/capture pairs
  EXPECT_EQ(runtime->info().active_sessions, 2u);
}

TEST_F(
    PrefixReuseTest,
    ContinuationExplicitOpenResetAndReplayHaveZeroCloneCalls) {
  start();
  submit(output(), {1, 2, 3}).wait();
  ASSERT_EQ(executor.clone_calls.load(), 1);
  auto continuation = output();
  submit(continuation, {1, 2, 3, 100, 4}).wait();
  ASSERT_TRUE(continuation->terminal);
  EXPECT_EQ(continuation->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto equal = output();
  submit(equal, {1, 2, 3, 100, 4, 101}).wait();
  EXPECT_EQ(equal->terminal->stats.session_reset_reason, "equal");
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto mismatch = output();
  submit(mismatch, {7, 8}).wait();
  EXPECT_EQ(mismatch->terminal->stats.session_reset_reason, "mismatch");
  EXPECT_EQ(executor.clone_calls.load(), 1);
  EXPECT_FALSE(runtime->reset_session_async("s").get());
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto reset = output();
  submit(reset, {1, 2, 3}).wait();
  EXPECT_EQ(reset->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
  EXPECT_FALSE(runtime->open_session_async("explicit").get());
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto explicit_open = output();
  submit(explicit_open, {1, 2, 3}, "explicit").wait();
  EXPECT_EQ(explicit_open->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
}

TEST_F(PrefixReuseTest, CaptureIsQueuedBeforeSubsequentDecode) {
  executor.reject_after_decode = true;
  start();
  submit(output(), {1, 2, 3}, "seed", 3).wait();
  auto snapshots = executor.clones();
  ASSERT_EQ(snapshots.size(), 1u);
  EXPECT_EQ(snapshots.front().prefix, (std::vector<Token>{1, 2, 3}));
  EXPECT_FALSE(runtime->close_session_async("seed").get());
  auto hit = output();
  submit(hit, {1, 2, 3}, "hit").wait();
  ASSERT_TRUE(hit->terminal);
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
}

TEST_F(PrefixReuseTest, SpeculativeTerminalCaptureContainsPromptOnly) {
  executor.burst = 3;
  start();
  submit(output(), {1, 2, 3, 4}, "seed", 3).wait();
  auto snapshots = executor.clones();
  ASSERT_EQ(snapshots.size(), 1u);
  EXPECT_EQ(snapshots.front().prefix, (std::vector<Token>{1, 2, 3, 4}));
  auto hit = output();
  submit(hit, {1, 2, 3, 4}, "hit", 3).wait();
  ASSERT_TRUE(hit->terminal);
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 3u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
  EXPECT_EQ(executor.feeds().back().tokens, (std::vector<Token>{4}));
}

TEST_F(
    PrefixReuseTest,
    BusyCaptureLaneSkipsLookupAndCaptureForOtherAdmissions) {
  start();
  auto slow = output();
  slow->block_first = true;
  slow->blocked.hold();
  auto handle = submit(slow, {1, 2, 3}, "seed");
  ASSERT_TRUE(slow->blocked.wait());
  std::vector<std::pair<std::shared_ptr<Events>, RequestHandle>> others;
  for (int i = 0; i < 3; ++i) {
    auto other = output();
    others.emplace_back(other, submit(other, {1, 2, 3}, std::nullopt));
  }
  // Process all admissions while the seed still owns the capture lane.
  ASSERT_FALSE(runtime->open_session_async("seed").get());
  EXPECT_EQ(executor.clone_calls.load(), 1);
  EXPECT_FALSE(handle.done());
  slow->blocked.release();
  handle.wait();
  for (auto& [other, other_handle] : others) {
    other_handle.wait();
    ASSERT_TRUE(other->terminal);
    EXPECT_EQ(other->terminal->stats.reused_prompt_tokens, 0u);
    EXPECT_EQ(other->terminal->stats.prefilled_prompt_tokens, 3u);
    EXPECT_EQ(executor.clone_calls.load(), 1);
  }
  auto hit = output();
  submit(hit, {1, 2, 3}, "hit").wait();
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
  EXPECT_EQ(executor.clone_calls.load(), 3);
}

TEST_F(PrefixReuseTest, SnapshotCapacityEvictsAndTransientRowsStayBounded) {
  config.max_sessions = 1;
  config.prefix_cache_capacity = 1;
  start();
  submit(output(), {1, 2, 3}, std::nullopt).wait();
  submit(output(), {7, 8, 9}, std::nullopt).wait();
  auto evicted = output();
  submit(evicted, {1, 2, 3}, std::nullopt).wait();
  ASSERT_TRUE(evicted->terminal);
  EXPECT_EQ(evicted->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 3); // all captures; no matching lookup
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_LE(executor.peak_rows.load(), 3);
}

TEST_F(
    PrefixReuseTest,
    UnsupportedCloneAndCaptureCapacityRefusalDoNotFailGeneration) {
  executor.refuse_clone = true;
  config.max_sessions = 1;
  executor.capacity = 1;
  start(false);
  auto first = output();
  submit(first, {1, 2, 3}, std::nullopt).wait();
  ASSERT_TRUE(first->terminal);
  EXPECT_EQ(first->terminal->finish_reason, FinishReason::Length);
  EXPECT_FALSE(first->terminal->error);
  executor.refuse_clone = false; // now clone is supported but lacks a spare row
  auto second = output();
  submit(second, {1, 2, 3}, std::nullopt).wait();
  ASSERT_TRUE(second->terminal);
  EXPECT_EQ(second->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(second->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(second->terminal->stats.prefilled_prompt_tokens, 3u);
  EXPECT_EQ(executor.clone_calls.load(), 2);
  EXPECT_TRUE(executor.clones().empty());
}

TEST_F(PrefixReuseTest, LookupRefusalFallsBackColdAndPreservesExistingEntry) {
  start();
  submit(output(), {1, 2, 3}, "seed").wait();
  executor.refuse_clone = true;
  auto cold = output();
  submit(cold, {1, 2, 3}, std::nullopt).wait();
  ASSERT_TRUE(cold->terminal);
  EXPECT_EQ(cold->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(cold->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(cold->terminal->stats.prefilled_prompt_tokens, 3u);
  executor.refuse_clone = false;
  auto hit = output();
  submit(hit, {1, 2, 3}, std::nullopt).wait();
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
}

TEST_F(
    PrefixReuseTest,
    CancelledPartialPrefillDoesNotCaptureOrSeedIncompletePrompt) {
  start();
  executor.executing.hold();
  std::vector<Token> prompt(40);
  std::iota(prompt.begin(), prompt.end(), 1);
  auto cancelled = output();
  auto handle = submit(cancelled, prompt);
  ASSERT_TRUE(executor.executing.wait());
  handle.cancel();
  executor.executing.release();
  handle.wait();
  ASSERT_TRUE(cancelled->terminal);
  EXPECT_EQ(cancelled->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(cancelled->terminal->stats.prefilled_prompt_tokens, 32u);
  EXPECT_EQ(executor.clone_calls.load(), 0);
  auto cold = output();
  submit(cold, prompt, "new").wait();
  EXPECT_EQ(cold->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(cold->terminal->stats.prefilled_prompt_tokens, 40u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
}

TEST_F(
    PrefixReuseTest,
    CancellationAfterCaptureDirtiesSourceButKeepsIndependentPromptSnapshot) {
  start();
  executor.decoding.hold();
  auto cancelled = output();
  auto handle = submit(cancelled, {1, 2, 3}, "s", 8);
  ASSERT_TRUE(executor.decoding.wait());
  handle.cancel();
  executor.decoding.release();
  handle.wait();
  ASSERT_TRUE(cancelled->terminal);
  EXPECT_EQ(cancelled->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_FALSE(cancelled->terminal->stats.generated_token_ids);
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto replay = output();
  submit(replay, {1, 2, 3, 100, 4}).wait();
  EXPECT_EQ(replay->terminal->stats.session_reset_reason, "dirty");
  EXPECT_EQ(replay->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto hit = output();
  submit(hit, {1, 2, 3}, "new").wait();
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
}

TEST_F(
    PrefixReuseTest,
    CancelledLookupReleasesCloneAndReservationWithoutGenerating) {
  start();
  submit(output(), {1, 2, 3}, "seed").wait();
  EXPECT_FALSE(runtime->close_session_async("seed").get());
  executor.cloning.hold();
  auto cancelled = output();
  auto handle = submit(cancelled, {1, 2, 3}, "lookup");
  ASSERT_TRUE(executor.cloning.wait());
  // Lookup can wait on the engine, but never holds the admission mutex.
  EXPECT_TRUE(runtime->info().ready);
  handle.cancel();
  executor.cloning.release();
  handle.wait();
  ASSERT_TRUE(cancelled->terminal);
  EXPECT_EQ(cancelled->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(cancelled->terminal->stats.prefilled_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 2);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  auto hit = output();
  submit(hit, {1, 2, 3}, "lookup").wait();
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
}

TEST_F(
    PrefixReuseTest,
    CloseBeforeCollectionDoesNotOverwriteExplicitReplacementHistory) {
  start();
  auto slow = output();
  auto replacement_events = output();
  slow->block_first = true;
  slow->blocked.hold();
  delivery_checkpoint.hold();
  slow->on_text = [this, replacement_events, count = 0]() mutable {
    if (++count == 4) {
      delivery_checkpoint.enter();
    }
    if (count == 5) {
      // The old capture cannot be collected until after this text turn.
      EXPECT_EQ(replacement_events->terminals, 1u);
    }
  };
  auto handle = submit(slow, {1, 2, 3}, "s", 6);
  ASSERT_TRUE(slow->blocked.wait());
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (executor.seen().size() < 6 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_EQ(executor.seen().size(), 6u);
  // Fence engine publication without collecting the old request's capture.
  ASSERT_FALSE(runtime->open_session_async("engine-barrier").get());
  ASSERT_FALSE(runtime->close_session_async("engine-barrier").get());
  EXPECT_FALSE(runtime->close_session_async("s").get());
  EXPECT_FALSE(runtime->open_session_async("s").get());
  executor.executing.hold();
  auto replacement = submit(replacement_events, {7, 8});
  ASSERT_TRUE(executor.executing.wait());
  executor.executing.release();
  ASSERT_FALSE(runtime->open_session_async("engine-barrier").get());
  EXPECT_EQ(executor.clone_calls.load(), 1);
  slow->blocked.release();
  ASSERT_TRUE(delivery_checkpoint.wait());
  // FIFO delivery has posted replacement preparation before old turn four.
  // Fence its control commit, then let its queued terminal precede turn five.
  ASSERT_FALSE(runtime->open_session_async("s").get());
  delivery_checkpoint.release();
  handle.wait();
  replacement.wait();
  EXPECT_EQ(slow->terminals, 1u);
  ASSERT_TRUE(slow->terminal);
  EXPECT_EQ(slow->terminal->stats.completion_tokens, 6u);
  ASSERT_TRUE(replacement_events->terminal);
  EXPECT_EQ(replacement_events->terminal->finish_reason, FinishReason::Length);
  auto continuation = output();
  submit(continuation, {7, 8, 100, 9}).wait();
  EXPECT_EQ(continuation->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(executor.clone_calls.load(), 1);
  auto hit = output();
  submit(hit, {1, 2, 3}, "new").wait();
  EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
}

TEST_F(
    PrefixReuseTest,
    ShutdownSettlesInFlightCaptureAndClosesEveryPhysicalRow) {
  start();
  executor.cloning.hold();
  auto event = output();
  auto handle = submit(event, {1, 2, 3});
  ASSERT_TRUE(executor.cloning.wait());
  auto stopped = std::async(std::launch::async, [&] { runtime->shutdown(); });
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (runtime->info().ready && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(runtime->info().ready);
  executor.cloning.release();
  EXPECT_EQ(stopped.wait_for(5s), std::future_status::ready);
  stopped.get();
  handle.wait();
  EXPECT_EQ(event->terminals, 1u);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.open_count(), 0);
}

TEST_F(PrefixReuseTest, ShutdownSettlesInFlightLookupAndCachedSnapshots) {
  start();
  submit(output(), {1, 2, 3}, "seed").wait();
  executor.cloning.hold();
  auto event = output();
  auto handle = submit(event, {1, 2, 3}, "lookup");
  ASSERT_TRUE(executor.cloning.wait());
  auto stopped = std::async(std::launch::async, [&] { runtime->shutdown(); });
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (runtime->info().ready && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(runtime->info().ready);
  executor.cloning.release();
  EXPECT_EQ(stopped.wait_for(5s), std::future_status::ready);
  stopped.get();
  handle.wait();
  ASSERT_TRUE(event->terminal);
  EXPECT_EQ(event->terminal->finish_reason, FinishReason::Cancelled);
  EXPECT_EQ(event->terminals, 1u);
  EXPECT_EQ(executor.open_count(), 0);
}

TEST_F(
    PrefixReuseTest,
    FailedPrefillDoesNotCaptureAndRecoveryReplayRemainsCold) {
  executor.fail_batches_from = 0;
  start();
  auto failed = output();
  submit(failed, {1, 2, 3}).wait();
  ASSERT_TRUE(failed->terminal);
  EXPECT_EQ(failed->terminal->finish_reason, FinishReason::Failed);
  EXPECT_TRUE(failed->terminal->error);
  EXPECT_EQ(failed->terminal->stats.prefilled_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 0);
  executor.fail_batches_from = -1;
  auto recovered = output();
  submit(recovered, {1, 2, 3}).wait();
  ASSERT_TRUE(recovered->terminal);
  EXPECT_EQ(recovered->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(recovered->terminal->stats.session_reset_reason, "dirty");
  EXPECT_EQ(recovered->terminal->stats.prefilled_prompt_tokens, 3u);
  EXPECT_EQ(executor.clone_calls.load(), 0);
  auto cold = output();
  submit(cold, {1, 2, 3}, "new").wait();
  ASSERT_TRUE(cold->terminal);
  EXPECT_EQ(cold->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(executor.clone_calls.load(), 1);
}

} // namespace

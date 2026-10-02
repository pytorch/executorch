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
#include <cstdlib>
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
    std::optional<batching::SamplingParams> sampling;
    std::optional<std::uint64_t> seed;
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
      feeds_.push_back(Feed{
          input.sid,
          static_cast<Position>(begin),
          std::move(fed),
          sampling_params(input.sid),
          sampling_seed(input.sid)});
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
  std::string text;
  std::optional<TerminalEvent> terminal;
  std::size_t terminals = 0;
  Gate blocked;
  Gate terminal_blocked;
  bool block_first = false;
  bool block_terminal = false;
  std::function<void()> on_text;
  void accept(GenerationEvent event) {
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

class PrefixReuseTest : public ::testing::Test {
 protected:
  Executor executor;
  Tokenizer tokenizer;
  ServingRuntimeConfig config;
  Gate delivery_checkpoint;
  Gate later_delivery_checkpoint;
  Gate cleanup_checkpoint;
  std::atomic<bool> captures_cleaned{false};
  std::promise<void> callbacks_finished;
  std::thread callback_watchdog;
  std::unique_ptr<ServingRuntime> runtime;
  std::vector<std::shared_ptr<Events>> events;
  void watch_callbacks() {
    callback_watchdog = std::thread([finished =
                                         callbacks_finished.get_future()] {
      if (finished.wait_for(30s) != std::future_status::ready) {
        ADD_FAILURE() << "prefix reuse callback test or teardown deadlocked";
        std::abort();
      }
    });
  }
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
    later_delivery_checkpoint.release();
    cleanup_checkpoint.release();
    executor.executing.release();
    executor.cloning.release();
    executor.decoding.release();
    executor.release();
    for (auto& event : events) {
      event->blocked.release();
      event->terminal_blocked.release();
    }
    runtime.reset();
    callbacks_finished.set_value();
    if (callback_watchdog.joinable()) {
      callback_watchdog.join();
    }
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
    return submit(event, std::move(tokens), std::move(key), options);
  }
  RequestHandle submit(
      const std::shared_ptr<Events>& event,
      std::vector<Token> tokens,
      std::optional<std::string> key,
      GenerationOptions options) {
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

class PrefixReuseSamplingTest
    : public PrefixReuseTest,
      public ::testing::WithParamInterface<std::pair<bool, bool>> {};

TEST_P(
    PrefixReuseSamplingTest,
    CaptureAndLookupApplyEachRequestsSamplingToFreshForward) {
  start();
  GenerationOptions seed_options;
  seed_options.max_new_tokens = 1;
  seed_options.sampling.temperature = GetParam().first ? 0.7f : 0.0f;
  seed_options.sampling.top_p = 0.8f;
  seed_options.sampling.top_k = 17;
  seed_options.seed = 123;
  auto seeded = output();
  submit(seeded, {1, 2, 3}, std::nullopt, seed_options).wait();
  ASSERT_TRUE(seeded->terminal);
  EXPECT_EQ(seeded->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(seeded->terminal->stats.reused_prompt_tokens, 0u);
  ASSERT_EQ(executor.clones().size(), 1u);
  EXPECT_EQ(executor.clones().front().prefix, (std::vector<Token>{1, 2, 3}));
  const auto seed_feed = executor.feeds().front();
  ASSERT_TRUE(seed_feed.sampling);
  EXPECT_FLOAT_EQ(
      seed_feed.sampling->temperature, seed_options.sampling.temperature);
  EXPECT_FLOAT_EQ(seed_feed.sampling->top_p, 0.8f);
  EXPECT_EQ(seed_feed.sampling->top_k, 17);
  EXPECT_EQ(seed_feed.seed, 123u);

  std::uint64_t next_seed = 456;
  for (auto key :
       {std::optional<std::string>{"named"}, std::optional<std::string>{}}) {
    GenerationOptions options;
    options.max_new_tokens = 1;
    options.sampling.temperature = GetParam().second ? 0.9f : 0.0f;
    options.sampling.top_p = 0.6f;
    options.sampling.top_k = 23;
    options.seed = next_seed++;
    auto hit = output();
    submit(hit, {1, 2, 3}, key, options).wait();
    ASSERT_TRUE(hit->terminal);
    EXPECT_EQ(hit->terminal->finish_reason, FinishReason::Length);
    EXPECT_FALSE(hit->terminal->error);
    EXPECT_EQ(hit->terminal->stats.reused_prompt_tokens, 2u);
    EXPECT_EQ(hit->terminal->stats.prefilled_prompt_tokens, 1u);
    const auto feed = executor.feeds().back();
    EXPECT_NE(feed.session, seed_feed.session);
    EXPECT_EQ(feed.position, 2);
    EXPECT_EQ(feed.tokens, (std::vector<Token>{3}));
    ASSERT_TRUE(feed.sampling);
    EXPECT_FLOAT_EQ(feed.sampling->temperature, options.sampling.temperature);
    EXPECT_FLOAT_EQ(feed.sampling->top_p, options.sampling.top_p);
    EXPECT_EQ(feed.sampling->top_k, options.sampling.top_k);
    EXPECT_EQ(feed.seed, options.seed);
    EXPECT_EQ(executor.seen().back().sampling_seed, options.seed);
  }
  // This fake verifies routing/configuration, not sampled model parity.
  EXPECT_FALSE(executor.executed_without_sampling_state());
  EXPECT_EQ(executor.clone_calls.load(), 5);
}

INSTANTIATE_TEST_SUITE_P(
    GreedyAndSampled,
    PrefixReuseSamplingTest,
    ::testing::Values(
        std::make_pair(false, false),
        std::make_pair(false, true),
        std::make_pair(true, false),
        std::make_pair(true, true)),
    [](const ::testing::TestParamInfo<std::pair<bool, bool>>& info) {
      return std::string(info.param.first ? "SampledSeed" : "GreedySeed") +
          (info.param.second ? "SampledLookup" : "GreedyLookup");
    });

TEST_F(PrefixReuseTest, DisabledCacheNeverCapturesOrLooksUpAnySamplingMode) {
  EXPECT_EQ(ServingRuntimeConfig{}.prefix_cache_capacity, 0u);
  config.prefix_cache_capacity = 0;
  start();
  for (float temperature : {0.0f, 0.7f}) {
    for (int request = 0; request < 2; ++request) {
      auto event = output();
      submit(event, {1, 2, 3}, std::nullopt, 1, temperature).wait();
      ASSERT_TRUE(event->terminal);
      EXPECT_EQ(event->terminal->finish_reason, FinishReason::Length);
      EXPECT_EQ(event->terminal->stats.reused_prompt_tokens, 0u);
      EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 3u);
      EXPECT_EQ(executor.clone_calls.load(), 0);
    }
  }
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
    BusyCaptureLaneWithUncollectedSeedMissesAndSkipsAdditionalCaptures) {
  watch_callbacks();
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

TEST_F(
    PrefixReuseTest,
    BusyCaptureLaneAllowsWarmLookupsWithoutLosingCaptureOwnership) {
  watch_callbacks();
  start();
  submit(output(), {1, 2, 3}, std::nullopt).wait();
  ASSERT_EQ(executor.clones().size(), 1u);
  const auto cached = executor.clones().front().destination;

  auto slow = output();
  slow->block_first = true;
  slow->blocked.hold();
  delivery_checkpoint.hold();
  later_delivery_checkpoint.hold();
  slow->on_text = [this, count = 0]() mutable {
    if (++count == 4) {
      delivery_checkpoint.enter();
    }
    if (count == 5) {
      later_delivery_checkpoint.enter();
    }
  };
  const auto owner = submit(slow, {7, 8, 9}, "owner", 6);
  ASSERT_TRUE(slow->blocked.wait());
  ASSERT_TRUE(wait_until([&] { return executor.feeds().size() == 7; }));
  EXPECT_EQ(executor.clone_calls.load(), 2); // Seed and owner's capture.

  auto first = output();
  const auto first_hit = submit(first, {1, 2, 3}, "first-hit");
  ASSERT_TRUE(wait_until([&] { return executor.feeds().size() == 8; }));
  // A new physical open fences engine publication before delivery resumes.
  auto barrier = runtime->open_session_async("engine-barrier");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(barrier.get());
  barrier = runtime->close_session_async("engine-barrier");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(barrier.get());
  EXPECT_EQ(executor.feeds().back().position, 2);
  EXPECT_EQ(executor.feeds().back().tokens, (std::vector<Token>{3}));
  EXPECT_EQ(executor.clone_calls.load(), 3); // Lookup, but no new capture.
  EXPECT_FALSE(owner.done());
  EXPECT_FALSE(first_hit.done());

  slow->blocked.release();
  ASSERT_TRUE(delivery_checkpoint.wait());
  // The short lookup has posted its finalizer before the owner's fourth
  // delivery turn. Control finalizes it and queues its terminal before turn 5.
  barrier = runtime->open_session_async("owner");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(barrier.get());
  delivery_checkpoint.release();
  ASSERT_TRUE(later_delivery_checkpoint.wait());
  ASSERT_TRUE(first_hit.done());
  ASSERT_TRUE(first->terminal);
  EXPECT_EQ(first->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(first->terminal->stats.prefilled_prompt_tokens, 1u);
  EXPECT_FALSE(owner.done());

  std::vector<std::pair<std::shared_ptr<Events>, RequestHandle>> later;
  for (auto key :
       {std::optional<std::string>{},
        std::optional<std::string>{"third-hit"}}) {
    auto event = output();
    later.emplace_back(event, submit(event, {1, 2, 3}, key));
    ASSERT_TRUE(wait_until(
        [&] { return executor.feeds().size() == 8 + later.size(); }));
    const auto feed = executor.feeds().back();
    EXPECT_EQ(feed.position, 2);
    EXPECT_EQ(feed.tokens, (std::vector<Token>{3}));
    EXPECT_FALSE(later.back().second.done());
    EXPECT_FALSE(owner.done());
    EXPECT_EQ(executor.clone_calls.load(), 3 + later.size());
  }
  const auto clones = executor.clones();
  ASSERT_EQ(clones.size(), 5u);
  EXPECT_EQ(clones[0].prefix, (std::vector<Token>{1, 2, 3}));
  EXPECT_EQ(clones[1].prefix, (std::vector<Token>{7, 8, 9}));
  for (std::size_t i = 2; i < clones.size(); ++i) {
    EXPECT_EQ(clones[i].source, cached);
    EXPECT_EQ(clones[i].prefix, (std::vector<Token>{1, 2}));
  }
  EXPECT_LE(executor.peak_rows.load(), 7); // 4 live + 2 cached + 1 capture.

  later_delivery_checkpoint.release();
  ASSERT_TRUE(wait_until([&] { return owner.done(); }));
  ASSERT_TRUE(slow->terminal);
  EXPECT_EQ(slow->terminal->finish_reason, FinishReason::Length);
  for (auto& [event, handle] : later) {
    ASSERT_TRUE(wait_until([&handle = handle] { return handle.done(); }));
    ASSERT_TRUE(event->terminal);
    EXPECT_EQ(event->terminal->finish_reason, FinishReason::Length);
    EXPECT_EQ(event->terminal->stats.reused_prompt_tokens, 2u);
    EXPECT_EQ(event->terminal->stats.prefilled_prompt_tokens, 1u);
  }
  EXPECT_EQ(executor.clone_calls.load(), 5);
  auto captured = output();
  const auto hit = submit(captured, {7, 8, 9}, std::nullopt);
  ASSERT_TRUE(wait_until([&] { return hit.done(); }));
  ASSERT_TRUE(captured->terminal);
  EXPECT_EQ(captured->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(captured->terminal->stats.prefilled_prompt_tokens, 1u);
  EXPECT_EQ(executor.clone_calls.load(), 7);
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

class PrefixReuseFenceTest : public PrefixReuseTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(
    PrefixReuseFenceTest,
    PublishedCaptureTextTerminalAndCleanupFenceExplicitReplacement) {
  watch_callbacks();
  start();
  auto slow = output();
  slow->block_first = true;
  slow->blocked.hold();
  slow->block_terminal = true;
  slow->terminal_blocked.hold();
  delivery_checkpoint.hold();
  slow->on_text = [this, count = 0]() mutable {
    if (++count == 4) {
      delivery_checkpoint.enter();
    }
  };
  auto capture = std::shared_ptr<int>(new int(0), [this](int* value) {
    cleanup_checkpoint.enter();
    delete value;
    captures_cleaned.store(true);
  });
  std::weak_ptr<int> retained = capture;
  GenerationOptions options;
  options.max_new_tokens = 6;
  auto result = runtime->generate(
      "s",
      PromptInput{{make_token_input({1, 2, 3})}},
      options,
      [slow, capture](GenerationEvent update) {
        (void)capture;
        slow->accept(std::move(update));
      });
  capture.reset();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  const auto old = std::get<RequestHandle>(result);
  ASSERT_TRUE(slow->blocked.wait());
  cleanup_checkpoint.hold();
  ASSERT_TRUE(wait_until([&] { return executor.feeds().size() == 6; }));
  // Fence the sixth engine publication while capture collection is still
  // waiting for the first public text callback to return.
  auto barrier = runtime->open_session_async("engine-barrier");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(barrier.get());
  barrier = runtime->close_session_async("engine-barrier");
  ASSERT_EQ(barrier.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(barrier.get());
  const auto clones = executor.clones();
  ASSERT_EQ(clones.size(), 1u);
  EXPECT_EQ(clones.front().prefix, (std::vector<Token>{1, 2, 3}));
  EXPECT_FALSE(retained.expired());

  auto ack = GetParam() ? runtime->reset_session_async("s")
                        : runtime->close_session_async("s");
  auto reopened = runtime->open_session_async("s");
  auto replacement_events = output();
  // This matches the old snapshot, but explicit open/reset must remain cold.
  const auto replacement = submit(replacement_events, {1, 2, 3, 7, 8});
  auto control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(control.get());
  auto other_events = output();
  const auto other = submit(other_events, {20, 21}, "other");
  ASSERT_TRUE(wait_until([&] { return executor.feeds().size() == 7; }));
  EXPECT_EQ(executor.feeds().back().tokens, (std::vector<Token>{20, 21}));
  const auto closed = executor.closed();
  EXPECT_NE(
      std::find(closed.begin(), closed.end(), clones.front().source),
      closed.end());
  EXPECT_EQ(
      std::find(closed.begin(), closed.end(), clones.front().destination),
      closed.end());
  const auto expect_pending = [&] {
    EXPECT_EQ(ack.wait_for(0s), std::future_status::timeout);
    EXPECT_EQ(reopened.wait_for(0s), std::future_status::timeout);
    EXPECT_FALSE(old.done());
    EXPECT_FALSE(replacement.done());
    EXPECT_FALSE(captures_cleaned.load());
    EXPECT_EQ(executor.feeds().size(), 7u);
    EXPECT_EQ(executor.clone_calls.load(), 1);
  };
  expect_pending();
  EXPECT_FALSE(other.done()); // Shared delivery is gated, engine is not.

  slow->blocked.release();
  ASSERT_TRUE(delivery_checkpoint.wait());
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(control.get());
  EXPECT_EQ(slow->text, "WXY");
  EXPECT_EQ(slow->terminals, 0u);
  expect_pending();
  delivery_checkpoint.release();
  ASSERT_TRUE(slow->terminal_blocked.wait());
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(control.get());
  ASSERT_TRUE(slow->terminal);
  EXPECT_EQ(slow->text, "WXYZAB");
  EXPECT_EQ(slow->terminals, 1u);
  EXPECT_EQ(slow->terminal->finish_reason, FinishReason::Length);
  EXPECT_FALSE(slow->terminal->error);
  EXPECT_EQ(slow->terminal->stats.completion_tokens, 6u);
  EXPECT_EQ(
      slow->terminal->stats.generated_token_ids,
      (std::vector<Token>{100, 101, 102, 103, 104, 105}));
  expect_pending();
  slow->terminal_blocked.release();
  ASSERT_TRUE(cleanup_checkpoint.wait());
  control = runtime->open_session_async("other");
  ASSERT_EQ(control.wait_for(5s), std::future_status::ready);
  ASSERT_FALSE(control.get());
  expect_pending();

  // Later engine work is outside the ACK cutoff and cannot delay it.
  executor.executing.hold();
  cleanup_checkpoint.release();
  ASSERT_EQ(ack.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_TRUE(old.done());
  EXPECT_TRUE(captures_cleaned.load());
  EXPECT_TRUE(retained.expired());
  ASSERT_EQ(reopened.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(reopened.get());
  ASSERT_TRUE(executor.executing.wait());
  EXPECT_FALSE(replacement.done());
  executor.executing.release();
  ASSERT_TRUE(wait_until([&] { return replacement.done() && other.done(); }));
  EXPECT_FALSE(old.error());
  EXPECT_FALSE(replacement.error());
  EXPECT_FALSE(other.error());
  ASSERT_TRUE(replacement_events->terminal);
  EXPECT_EQ(replacement_events->terminal->finish_reason, FinishReason::Length);
  EXPECT_EQ(replacement_events->terminal->stats.session_reset_reason, "new");
  EXPECT_EQ(replacement_events->terminal->stats.reused_prompt_tokens, 0u);
  EXPECT_EQ(replacement_events->terminal->stats.prefilled_prompt_tokens, 5u);
  EXPECT_EQ(executor.feeds().back().position, 0);
  EXPECT_EQ(
      executor.feeds().back().tokens, (std::vector<Token>{1, 2, 3, 7, 8}));
  EXPECT_EQ(executor.clone_calls.load(), 1);

  auto continued = output();
  const auto continuation = submit(continued, {1, 2, 3, 7, 8, 100, 9});
  ASSERT_TRUE(wait_until([&] { return continuation.done(); }));
  ASSERT_TRUE(continued->terminal);
  EXPECT_EQ(continued->terminal->stats.session_reset_reason, "exact_prefix");
  EXPECT_EQ(continued->terminal->stats.reused_prompt_tokens, 5u);
  EXPECT_EQ(continued->terminal->stats.prefilled_prompt_tokens, 2u);
  EXPECT_EQ(executor.feeds().back().position, 5);
  EXPECT_EQ(executor.feeds().back().tokens, (std::vector<Token>{100, 9}));
  EXPECT_EQ(executor.clone_calls.load(), 1);
  // Closing/resetting the source did not retire its independent snapshot.
  auto hit_events = output();
  const auto hit = submit(hit_events, {1, 2, 3}, "new");
  ASSERT_TRUE(wait_until([&] { return hit.done(); }));
  ASSERT_TRUE(hit_events->terminal);
  EXPECT_EQ(hit_events->terminal->stats.reused_prompt_tokens, 2u);
  EXPECT_EQ(hit_events->terminal->stats.prefilled_prompt_tokens, 1u);
  const auto final_clones = executor.clones();
  ASSERT_EQ(final_clones.size(), 3u);
  EXPECT_EQ(final_clones[1].source, clones.front().destination);
  EXPECT_EQ(final_clones[1].prefix, (std::vector<Token>{1, 2}));
  EXPECT_LE(executor.peak_rows.load(), 7);
}

INSTANTIATE_TEST_SUITE_P(
    CloseAndReset,
    PrefixReuseFenceTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Reset" : "Close";
    });

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

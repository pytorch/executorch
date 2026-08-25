/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::Runner;
using executorch::extension::llm::batching::DecodeFirstScheduler;
using executorch::extension::llm::batching::FinishReason;
using executorch::extension::llm::batching::GenConfig;
using executorch::extension::llm::batching::GenerationHandle;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::RunnerConfig;
using executorch::extension::llm::batching::Scheduler;
using executorch::extension::llm::batching::Session;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::Task;
using executorch::extension::llm::batching::Token;
using executorch::extension::llm::batching::testing::FakeDFlashExecutor;
using executorch::extension::llm::batching::testing::FakeExecutor;

namespace {

constexpr std::chrono::seconds kTimeout{5};

class Updates {
 public:
  void operator()(
      const std::vector<Token>& update,
      std::optional<FinishReason> finish) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tokens_.insert(tokens_.end(), update.begin(), update.end());
      if (finish) {
        finish_ = *finish;
        terminal_calls_++;
      }
    }
    cv_.notify_all();
  }

  bool wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, kTimeout, [this] { return finish_.has_value(); });
  }

  std::vector<Token> tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tokens_;
  }

  std::optional<FinishReason> finish() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return finish_;
  }

  int terminal_calls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return terminal_calls_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<Token> tokens_;
  std::optional<FinishReason> finish_;
  int terminal_calls_ = 0;
};

struct Fixture {
  explicit Fixture(
      FakeExecutor& executor,
      std::int32_t max_decode_sequences = 4,
      std::int32_t max_prefill_chunk_size = 8,
      RunnerConfig runner_config = RunnerConfig{})
      : scheduler(DecodeFirstScheduler::create(
            2 * max_prefill_chunk_size + max_decode_sequences,
            max_decode_sequences,
            static_cast<std::size_t>(max_prefill_chunk_size))),
        runner(executor, *scheduler, runner_config) {}

  std::unique_ptr<DecodeFirstScheduler> scheduler;
  Runner runner;
};

Session open(Runner& runner) {
  auto future = runner.open_session();
  EXPECT_EQ(future.wait_for(kTimeout), std::future_status::ready);
  auto session = future.get();
  EXPECT_TRUE(session.has_value());
  return session.value_or(Session{});
}

std::vector<Token> tokens(int n, Token first = 100) {
  std::vector<Token> result;
  result.reserve(n);
  for (int i = 0; i < n; ++i) {
    result.push_back(first + i);
  }
  return result;
}

GenConfig config(std::int32_t max_new_tokens) {
  GenConfig result;
  result.max_new_tokens = max_new_tokens;
  return result;
}

GenerationHandle generate(
    const Session& session,
    std::vector<Token> delta,
    GenConfig generation_config,
    const std::shared_ptr<Updates>& updates) {
  return session.generate_async(
      std::move(delta),
      std::move(generation_config),
      [updates](
          const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) { (*updates)(emitted, finish); });
}

class RejectingScheduler : public Scheduler {
 public:
  bool submit(std::vector<Task>) override {
    return false;
  }
  bool has_work() const override {
    return false;
  }
  std::vector<Task> get_work() override {
    return {};
  }
  std::vector<Task> cancel(SessionId) override {
    return {};
  }
  std::vector<Task> clear() override {
    return {};
  }
};

} // namespace

TEST(SessionTest, OpenAndCloseRoundTripThroughTheEngineThread) {
  FakeExecutor executor;
  Fixture fixture(executor);

  Session first = open(fixture.runner);
  Session second = open(fixture.runner);
  EXPECT_NE(first.id(), second.id());
  EXPECT_EQ(executor.open_count(), 2);

  auto closed = first.close();
  ASSERT_EQ(closed.wait_for(kTimeout), std::future_status::ready);
  closed.get();
  EXPECT_EQ(executor.open_count(), 1);
  EXPECT_EQ(executor.closed(), (std::vector<SessionId>{first.id()}));
}

TEST(SessionTest, CapacityRefusalIsReported) {
  FakeExecutor executor;
  executor.capacity = 1;
  Fixture fixture(executor);

  EXPECT_TRUE(open(fixture.runner).valid());
  auto refused = fixture.runner.open_session();
  ASSERT_EQ(refused.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(refused.get().has_value());
}

TEST(SessionTest, ClosingCopiedSessionTwiceClosesExecutorOnce) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  Session copy = session;

  session.close().get();
  copy.close().get();

  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed(), (std::vector<SessionId>{session.id()}));
}

TEST(SessionTest, ClosingUnknownSessionIsANoOp) {
  FakeExecutor executor;
  Fixture fixture(executor);

  fixture.runner.close_session(999).get();

  EXPECT_TRUE(executor.closed().empty());
}

TEST(SessionTest, CopiedSessionCannotGenerateAfterClose) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  Session stale = session;
  session.close().get();
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(stale, tokens(2), config(1), updates);

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_FALSE(executor.has_sampling_state(session.id()));
  EXPECT_TRUE(executor.seen().empty());
}

TEST(SessionTest, FabricatedSessionCannotGenerate) {
  FakeExecutor executor;
  Fixture fixture(executor);
  constexpr SessionId kUnknownSession = 999;
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = fixture.runner.generate_async(
      kUnknownSession,
      tokens(2),
      config(1),
      [updates](
          const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) { (*updates)(emitted, finish); });

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_FALSE(executor.has_sampling_state(kUnknownSession));
  EXPECT_TRUE(executor.seen().empty());
}

TEST(SessionTest, CloseQueuedBeforeStartRejectsStart) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session blocker = open(fixture.runner);
  Session target = open(fixture.runner);
  auto blocker_updates = std::make_shared<Updates>();
  GenerationHandle blocker_handle =
      generate(blocker, tokens(2), config(100000), blocker_updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto closed = target.close();
  auto target_updates = std::make_shared<Updates>();
  GenerationHandle target_handle =
      generate(target, tokens(2), config(1), target_updates);
  executor.release();

  ASSERT_EQ(closed.wait_for(kTimeout), std::future_status::ready);
  closed.get();
  ASSERT_TRUE(target_updates->wait());
  EXPECT_EQ(target_updates->finish(), FinishReason::Failed);
  EXPECT_FALSE(executor.has_sampling_state(target.id()));
  for (const FakeExecutor::Seen& seen : executor.seen()) {
    EXPECT_NE(seen.session, target.id());
  }

  blocker_handle.cancel();
  ASSERT_TRUE(blocker_updates->wait());
}

TEST(GenerationTest, SeedIsSetBeforeTasksExecute) {
  FakeExecutor executor;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();
  GenConfig generation_config = config(1);
  generation_config.seed = 1234;

  Session session = open(fixture.runner);
  GenerationHandle handle =
      generate(session, tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_TRUE(executor.has_sampling_state(session.id()));
  EXPECT_EQ(executor.sampling_seed(session.id()), 1234u);
  EXPECT_FALSE(executor.seen().empty());
  EXPECT_FALSE(executor.executed_without_sampling_state());
}

TEST(GenerationTest, TerminalStateIsPublishedBeforeCallback) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  std::atomic<bool> done_visible{false};
  std::atomic<bool> reason_visible{false};
  GenerationHandle handle;

  handle = session.generate_async(
      tokens(2),
      config(1),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        if (finish) {
          done_visible.store(handle.done());
          reason_visible.store(handle.finish_reason() == *finish);
        }
        (*updates)(emitted, finish);
      });
  executor.release();

  ASSERT_TRUE(updates->wait());
  EXPECT_TRUE(done_visible.load());
  EXPECT_TRUE(reason_visible.load());
  EXPECT_EQ(updates->terminal_calls(), 1);
}

TEST(GenerationTest, NullSeedInitializesNondeterministicSamplingState) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(session, tokens(2), config(1), updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_TRUE(executor.has_sampling_state(session.id()));
  EXPECT_FALSE(executor.sampling_seed(session.id()).has_value());
  EXPECT_FALSE(executor.executed_without_sampling_state());
}

TEST(GenerationTest, ActiveGenerationRejectsDuplicateWithoutReseeding) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto active_updates = std::make_shared<Updates>();
  GenConfig active_config = config(2);
  active_config.seed = 12;
  GenerationHandle active =
      generate(session, tokens(2), active_config, active_updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto duplicate_updates = std::make_shared<Updates>();
  GenConfig duplicate_config = config(1);
  duplicate_config.seed = 37;
  GenerationHandle duplicate =
      generate(session, tokens(1), duplicate_config, duplicate_updates);
  executor.release();

  ASSERT_TRUE(duplicate_updates->wait());
  ASSERT_TRUE(active_updates->wait());
  EXPECT_EQ(duplicate_updates->finish(), FinishReason::Failed);
  EXPECT_EQ(active_updates->finish(), FinishReason::NewTokenLimit);
  EXPECT_EQ(executor.sampling_seed(session.id()), 12u);
  for (const FakeExecutor::Seen& seen : executor.seen()) {
    EXPECT_EQ(seen.sampling_seed, 12u);
  }
}

TEST(GenerationTest, ANewGenerationReseedsTheExistingSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto first_updates = std::make_shared<Updates>();
  GenConfig first_config = config(1);
  first_config.seed = 11;
  GenerationHandle first =
      generate(session, tokens(2), first_config, first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(executor.sampling_seed(session.id()), 11u);
  const std::vector<Token> first_tokens = first_updates->tokens();
  ASSERT_EQ(first_tokens.size(), 1u);
  std::mt19937_64 first_rng(11);
  EXPECT_EQ(first_tokens[0], session.id() * 1000 + first_rng() % 1000);

  auto second_updates = std::make_shared<Updates>();
  GenConfig second_config = config(1);
  second_config.seed = 22;
  GenerationHandle second =
      generate(session, tokens(1), second_config, second_updates);
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(executor.sampling_seed(session.id()), 22u);
  const std::vector<Token> second_tokens = second_updates->tokens();
  ASSERT_EQ(second_tokens.size(), 1u);
  std::mt19937_64 second_rng(22);
  EXPECT_EQ(second_tokens[0], session.id() * 1000 + second_rng() % 1000);
}

// A session opens at position 0 and the caller never names a position, so the
// first slice of a fresh session always lands there.
TEST(DeltaTest, FreshSessionStartsAtZero) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(3), config(1), updates);
  ASSERT_TRUE(updates->wait());
  handle.wait();

  ASSERT_EQ(updates->finish(), FinishReason::NewTokenLimit);
  ASSERT_EQ(executor.seen().size(), 1u);
  EXPECT_EQ(executor.seen()[0].position, 0);
  EXPECT_EQ(executor.seen()[0].offset, 0u);
  EXPECT_EQ(executor.seen()[0].effective_position(), 0);
}

TEST(DeltaTest, ChunksUseDeltaBasePlusOffset) {
  FakeExecutor executor;
  RunnerConfig runner_config;
  runner_config.max_prefill_chunk_size = 4;
  Fixture fixture(executor, 2, 4, runner_config);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(open(fixture.runner), tokens(10), config(1), updates);
  ASSERT_TRUE(updates->wait());
  handle.wait();

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_EQ(seen.size(), 3u);
  EXPECT_EQ(seen[0].position, 0);
  EXPECT_EQ(seen[0].offset, 0u);
  EXPECT_EQ(seen[0].size, 4u);
  EXPECT_EQ(seen[0].effective_position(), 0);
  EXPECT_EQ(seen[1].position, 0);
  EXPECT_EQ(seen[1].offset, 4u);
  EXPECT_EQ(seen[1].size, 4u);
  EXPECT_EQ(seen[1].effective_position(), 4);
  EXPECT_EQ(seen[2].position, 0);
  EXPECT_EQ(seen[2].offset, 8u);
  EXPECT_EQ(seen[2].size, 2u);
  EXPECT_EQ(seen[2].effective_position(), 8);
}

// The session owns its position, so a later turn appends rather than
// restarting. The caller supplies only the new tokens.
TEST(DeltaTest, SecondTurnContinuesWhereTheFirstEnded) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto first_updates = std::make_shared<Updates>();
  GenerationHandle first =
      generate(session, tokens(3), config(2), first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(first_updates->finish(), FinishReason::NewTokenLimit);

  const Position after_first = executor.seen().back().effective_position() +
      static_cast<Position>(executor.seen().back().size);

  auto second_updates = std::make_shared<Updates>();
  GenerationHandle second =
      generate(session, std::vector<Token>{500}, config(1), second_updates);
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(second_updates->finish(), FinishReason::NewTokenLimit);

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_GE(seen.size(), 3u);
  EXPECT_EQ(seen.back().size, 1u);
  EXPECT_GE(seen.back().effective_position(), after_first - 1)
      << "the second turn must append, not restart at 0";
  EXPECT_GT(seen.back().effective_position(), 0);
}

// Each open_session() gets its own position, so one session's progress does
// not move another.
TEST(DeltaTest, PositionIsPerSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session first = open(fixture.runner);
  Session second = open(fixture.runner);

  auto first_updates = std::make_shared<Updates>();
  generate(first, tokens(4), config(1), first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(first_updates->finish(), FinishReason::NewTokenLimit);

  auto second_updates = std::make_shared<Updates>();
  generate(second, tokens(2), config(1), second_updates);
  ASSERT_TRUE(second_updates->wait());
  ASSERT_EQ(second_updates->finish(), FinishReason::NewTokenLimit);

  for (const FakeExecutor::Seen& slice : executor.seen()) {
    if (slice.session == second.id() && slice.size == 2u) {
      EXPECT_EQ(slice.effective_position(), 0)
          << "the second session must open at 0";
    }
  }
}

TEST(DeltaTest, GeneratedTokenUsesExecutorContinuationPosition) {
  FakeExecutor executor;
  executor.continuation_position = 101;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(open(fixture.runner), tokens(2), config(2), updates);
  ASSERT_TRUE(updates->wait());

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0].effective_position(), 0);
  EXPECT_EQ(seen[1].effective_position(), 101);
}

// The caller no longer names a position, so the only rejectable input left is
// the delta itself and the token budget.
TEST(DeltaTest, InvalidInitialRangesFailBeforeBeginning) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto empty_delta = std::make_shared<Updates>();
  generate(session, {}, config(1), empty_delta);
  ASSERT_TRUE(empty_delta->wait());
  EXPECT_EQ(empty_delta->finish(), FinishReason::Failed);
  EXPECT_EQ(empty_delta->terminal_calls(), 1);

  auto bad_budget = std::make_shared<Updates>();
  generate(session, tokens(1), config(0), bad_budget);
  ASSERT_TRUE(bad_budget->wait());
  EXPECT_EQ(bad_budget->finish(), FinishReason::Failed);
  EXPECT_EQ(bad_budget->terminal_calls(), 1);

  EXPECT_FALSE(executor.has_sampling_state(session.id()));
  EXPECT_TRUE(executor.seen().empty());
}

TEST(DeltaTest, MalformedContinuationFailsWithoutAnotherTask) {
  for (int malformed = 0; malformed < 4; ++malformed) {
    FakeExecutor executor;
    if (malformed == 0) {
      executor.omit_continuation = true;
    } else if (malformed == 1) {
      executor.null_continuation_tokens = true;
    } else if (malformed == 2) {
      executor.empty_continuation = true;
    } else {
      executor.continuation_position = -1;
    }
    Fixture fixture(executor);
    auto updates = std::make_shared<Updates>();

    GenerationHandle handle =
        generate(open(fixture.runner), tokens(2), config(2), updates);
    ASSERT_TRUE(updates->wait());
    EXPECT_EQ(updates->finish(), FinishReason::Failed);
    EXPECT_EQ(updates->terminal_calls(), 1);
    EXPECT_EQ(executor.seen().size(), 1u);
  }
}

TEST(GenerationTest, StopTokenAndBudgetAreAppliedToUpdates) {
  FakeExecutor executor;
  executor.stop_token = 999;
  executor.emit_before_stop = 2;
  Fixture fixture(executor);
  GenConfig generation_config = config(50);
  generation_config.stop_tokens = {999};
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(
      open(fixture.runner), tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::StopToken);
  const std::vector<Token> emitted = updates->tokens();
  EXPECT_EQ(emitted.size(), 2u);
  EXPECT_EQ(std::find(emitted.begin(), emitted.end(), 999), emitted.end());
}

TEST(SpeculativeTest, AcceptedRunUsesExecutorPosition) {
  FakeDFlashExecutor executor;
  executor.n_draft = 2;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(open(fixture.runner), tokens(2), config(7), updates);
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::NewTokenLimit);
  EXPECT_EQ(updates->tokens().size(), 7u);

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_GE(seen.size(), 3u);
  EXPECT_EQ(seen[0].effective_position(), 0);
  EXPECT_EQ(seen[1].effective_position(), 2);
  EXPECT_EQ(seen[2].effective_position(), 5);
}

TEST(FailureTest, SchedulerRejectionEndsTheGeneration) {
  FakeExecutor executor;
  RejectingScheduler scheduler;
  Runner runner(executor, scheduler, RunnerConfig{});
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(open(runner), tokens(2), config(2), updates);
  ASSERT_TRUE(updates->wait());
  handle.wait();
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_TRUE(executor.seen().empty());
}

TEST(FailureTest, FailureOnLeadingPrefillChunksEndsGeneration) {
  FakeExecutor executor;
  executor.fail_batches_from = 0;
  RunnerConfig runner_config;
  runner_config.max_prefill_chunk_size = 4;
  Fixture fixture(executor, 1, 4, runner_config);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(open(fixture.runner), tokens(10), config(1), updates);

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0].size, 4u);
  EXPECT_EQ(seen[1].size, 4u);
}

TEST(FailureTest, ExecutorFailurePoisonsTheSession) {
  FakeExecutor executor;
  executor.fail_batches_from = 0;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto first_updates = std::make_shared<Updates>();
  GenerationHandle first =
      generate(session, tokens(2), config(2), first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(first_updates->finish(), FinishReason::Failed);

  executor.fail_batches_from = -1;
  auto second_updates = std::make_shared<Updates>();
  GenerationHandle second =
      generate(session, tokens(1), config(1), second_updates);
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(second_updates->finish(), FinishReason::Failed);
}

TEST(CancelTest, ExplicitCancellationEndsTheGeneration) {
  FakeExecutor executor;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(open(fixture.runner), tokens(2), config(100000), updates);

  handle.cancel();
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
}

// The session tracks what the executor consumed, not what the runner asked
// for, so a cancelled generation leaves it where the executor actually stopped
// rather than back at the position its first chunk started from.
TEST(CancelTest, PositionSurvivesACancelledGeneration) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto cancelled = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(4), config(100000), cancelled);
  // Let it consume the prompt and emit before cutting it short.
  while (cancelled->tokens().empty()) {
    std::this_thread::yield();
  }
  handle.cancel();
  ASSERT_TRUE(cancelled->wait());
  ASSERT_EQ(cancelled->finish(), FinishReason::Cancelled);

  const std::size_t consumed = executor.seen().size();
  ASSERT_GT(consumed, 0u);

  auto resumed = std::make_shared<Updates>();
  generate(session, tokens(1), config(1), resumed);
  ASSERT_TRUE(resumed->wait());
  ASSERT_EQ(resumed->finish(), FinishReason::NewTokenLimit);

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_GT(seen.size(), consumed);
  EXPECT_GE(seen.back().effective_position(), 4)
      << "the resumed turn must start past the prompt the executor consumed, "
         "not back at 0";
}

TEST(SessionTest, ClosingASessionEndsItsGeneration) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(2), config(100000), updates);

  session.close().get();
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
}

TEST(ShutdownTest, EndsLiveAndRejectsNewWork) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto live_updates = std::make_shared<Updates>();
  GenerationHandle live =
      generate(session, tokens(2), config(100000), live_updates);

  fixture.runner.shutdown();
  ASSERT_TRUE(live_updates->wait());
  EXPECT_EQ(live_updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed(), (std::vector<SessionId>{session.id()}));

  auto opened = fixture.runner.open_session();
  ASSERT_EQ(opened.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(opened.get().has_value());

  auto closed = fixture.runner.close_session(session.id());
  ASSERT_EQ(closed.wait_for(kTimeout), std::future_status::ready);

  auto rejected_updates = std::make_shared<Updates>();
  const std::thread::id caller_thread = std::this_thread::get_id();
  std::thread::id callback_thread;
  GenerationHandle rejected = session.generate_async(
      tokens(1),
      config(1),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        callback_thread = std::this_thread::get_id();
        (*rejected_updates)(emitted, finish);
      });

  EXPECT_EQ(callback_thread, caller_thread);
  EXPECT_TRUE(rejected.done());
  EXPECT_EQ(rejected.finish_reason(), FinishReason::Cancelled);
  ASSERT_TRUE(rejected_updates->wait());
  EXPECT_TRUE(rejected_updates->tokens().empty());
  EXPECT_EQ(rejected_updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(rejected_updates->terminal_calls(), 1);
}

TEST(ShutdownTest, DiscardsInFlightOutputAndCancelsGeneration) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(2), config(1), updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  std::thread stopping([&] { fixture.runner.shutdown(); });
  std::vector<std::future<std::optional<Session>>> admitted_opens;
  while (true) {
    auto opened = fixture.runner.open_session();
    if (opened.wait_for(std::chrono::milliseconds(1)) ==
        std::future_status::ready) {
      EXPECT_FALSE(opened.get().has_value());
      break;
    }
    admitted_opens.push_back(std::move(opened));
  }
  executor.release();
  stopping.join();

  for (auto& opened : admitted_opens) {
    ASSERT_EQ(opened.wait_for(kTimeout), std::future_status::ready);
    EXPECT_FALSE(opened.get().has_value());
  }
  ASSERT_TRUE(updates->wait());
  EXPECT_TRUE(updates->tokens().empty());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_EQ(executor.seen().size(), 1u);
  EXPECT_EQ(executor.open_count(), 0);
}

TEST(ShutdownTest, ConcurrentCallersWaitForFullStop) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(2), config(2), updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto owner = std::async(std::launch::async, [&] { fixture.runner.shutdown(); });
  while (true) {
    auto opened = fixture.runner.open_session();
    if (opened.wait_for(std::chrono::milliseconds(1)) ==
        std::future_status::ready) {
      EXPECT_FALSE(opened.get().has_value());
      break;
    }
  }

  constexpr int kWaiters = 3;
  std::vector<std::future<void>> waiters;
  for (int i = 0; i < kWaiters; ++i) {
    waiters.push_back(
        std::async(std::launch::async, [&] { fixture.runner.shutdown(); }));
  }
  EXPECT_EQ(owner.wait_for(std::chrono::milliseconds(20)),
            std::future_status::timeout);
  for (auto& waiter : waiters) {
    EXPECT_EQ(
        waiter.wait_for(std::chrono::milliseconds(20)),
        std::future_status::timeout);
  }

  executor.release();
  ASSERT_EQ(owner.wait_for(kTimeout), std::future_status::ready);
  for (auto& waiter : waiters) {
    ASSERT_EQ(waiter.wait_for(kTimeout), std::future_status::ready);
  }

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed(), (std::vector<SessionId>{session.id()}));
}

TEST(ShutdownTest, CallbackShutdownStopsLaterOutputsInTheSameBatch) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session barrier = open(fixture.runner);
  Session first = open(fixture.runner);
  Session second = open(fixture.runner);
  auto barrier_updates = std::make_shared<Updates>();
  auto first_updates = std::make_shared<Updates>();
  auto second_updates = std::make_shared<Updates>();

  GenerationHandle barrier_handle =
      generate(barrier, tokens(2), config(1), barrier_updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }
  std::atomic<bool> requested{false};
  GenerationHandle first_handle = first.generate_async(
      tokens(2),
      config(2),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        (*first_updates)(emitted, finish);
        if (!finish && !requested.exchange(true)) {
          fixture.runner.shutdown();
        }
      });
  GenerationHandle second_handle =
      generate(second, tokens(2), config(2), second_updates);
  executor.release();

  while (!requested.load()) {
    std::this_thread::yield();
  }
  fixture.runner.shutdown();

  ASSERT_TRUE(barrier_updates->wait());
  ASSERT_TRUE(first_updates->wait());
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(first_updates->tokens().size(), 1u);
  EXPECT_TRUE(second_updates->tokens().empty());
  EXPECT_EQ(first_updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(second_updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(first_updates->terminal_calls(), 1);
  EXPECT_EQ(second_updates->terminal_calls(), 1);
  EXPECT_EQ(executor.batch_sizes(), (std::vector<int>{1, 2}));
}

TEST(ShutdownTest, CallbackCanRequestShutdown) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  std::atomic<bool> requested{false};

  GenerationHandle handle = session.generate_async(
      tokens(2),
      config(2),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        (*updates)(emitted, finish);
        if (!finish && !requested.exchange(true)) {
          fixture.runner.shutdown();
        }
      });

  while (!requested.load()) {
    std::this_thread::yield();
  }
  fixture.runner.shutdown();

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->tokens().size(), 1u);
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_EQ(executor.seen().size(), 1u);
  EXPECT_EQ(executor.open_count(), 0);
}

TEST(ShutdownTest, ClosesRemainingSessionsExactlyOnce) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session first = open(fixture.runner);
  Session second = open(fixture.runner);
  Session third = open(fixture.runner);
  second.close().get();

  fixture.runner.shutdown();

  EXPECT_EQ(executor.open_count(), 0);
  std::vector<SessionId> closed = executor.closed();
  std::sort(closed.begin(), closed.end());
  EXPECT_EQ(
      closed,
      (std::vector<SessionId>{first.id(), second.id(), third.id()}));
}

TEST(ShutdownTest, ConcurrentAdmissionDoesNotStrandCallers) {
  FakeExecutor executor;
  executor.capacity = 64;
  Fixture fixture(executor);
  std::vector<Session> sessions;
  for (int i = 0; i < 16; ++i) {
    sessions.push_back(open(fixture.runner));
  }

  std::atomic<bool> go{false};
  std::atomic<int> stranded{0};
  std::vector<std::thread> callers;
  for (int thread = 0; thread < 4; ++thread) {
    callers.emplace_back([&, thread] {
      while (!go.load()) {
        std::this_thread::yield();
      }
      for (int i = thread; i < static_cast<int>(sessions.size()); i += 4) {
        auto updates = std::make_shared<Updates>();
        GenerationHandle handle =
            generate(sessions[i], tokens(2), config(2), updates);
        if (!updates->wait()) {
          stranded++;
          return;
        }
      }
    });
  }

  go.store(true);
  std::this_thread::yield();
  fixture.runner.shutdown();
  for (std::thread& caller : callers) {
    caller.join();
  }
  EXPECT_EQ(stranded.load(), 0);
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>
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
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::DecodeFirstScheduler;
using executorch::extension::llm::batching::Executor;
using executorch::extension::llm::batching::FinishReason;
using executorch::extension::llm::batching::GenConfig;
using executorch::extension::llm::batching::GenerationHandle;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::Runner;
using executorch::extension::llm::batching::SamplingParams;
using executorch::extension::llm::batching::Scheduler;
using executorch::extension::llm::batching::Session;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::Task;
using executorch::extension::llm::batching::Token;
using executorch::extension::llm::batching::testing::FakeExecutor;

namespace {

constexpr std::chrono::seconds kTimeout{5};

static_assert(std::is_move_constructible<Session>::value);
static_assert(std::is_move_assignable<Session>::value);
static_assert(!std::is_copy_constructible<Session>::value);
static_assert(!std::is_copy_assignable<Session>::value);

class ExecutorWithoutSampling : public Executor {
 public:
  std::optional<SessionId> open_session() override {
    return std::nullopt;
  }

  void close_session(SessionId) override {}

  bool execute(
      const executorch::extension::llm::batching::BatchInput&,
      executorch::extension::llm::batching::BatchOutput&) override {
    return false;
  }
};

static_assert(std::is_abstract<ExecutorWithoutSampling>::value);

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
      std::int32_t max_prefill_chunk_size = 8)
      : scheduler(DecodeFirstScheduler::create(
            2 * max_prefill_chunk_size + max_decode_sequences,
            max_decode_sequences,
            static_cast<std::size_t>(max_prefill_chunk_size))),
        runner(executor, *scheduler) {}

  std::unique_ptr<DecodeFirstScheduler> scheduler;
  Runner runner;
};

Session open(Runner& runner) {
  auto future = runner.open_session();
  // Not ASSERT_: this helper returns a value, so a fatal assertion cannot
  // early-return from it. Bail explicitly, or the get() below would block
  // forever on exactly the bug the timeout is here to catch.
  if (future.wait_for(kTimeout) != std::future_status::ready) {
    ADD_FAILURE() << "open_session did not settle within the timeout";
    return Session{};
  }
  auto session = future.get();
  EXPECT_TRUE(session.has_value());
  return session ? std::move(*session) : Session{};
}

SessionId last_opened(const FakeExecutor& executor) {
  const std::vector<SessionId> opened = executor.opened();
  EXPECT_FALSE(opened.empty());
  return opened.empty() ? SessionId{} : opened.back();
}

bool wait_for_open_count(const FakeExecutor& executor, int expected) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (executor.open_count() == expected) {
      return true;
    }
    std::this_thread::yield();
  }
  return executor.open_count() == expected;
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
  std::size_t max_prefill_chunk_size() const override {
    return 8;
  }
  std::vector<Task> clear() override {
    return {};
  }
};

} // namespace

TEST(SessionTest, DefaultSessionRejectsGenerationSynchronously) {
  Session session;
  EXPECT_FALSE(session.valid());
  auto updates = std::make_shared<Updates>();
  const std::thread::id caller_thread = std::this_thread::get_id();
  std::thread::id callback_thread;

  GenerationHandle handle = session.generate_async(
      tokens(1),
      config(1),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        callback_thread = std::this_thread::get_id();
        (*updates)(emitted, finish);
      });

  EXPECT_EQ(callback_thread, caller_thread);
  EXPECT_TRUE(handle.done());
  EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
  ASSERT_TRUE(updates->wait());
  EXPECT_TRUE(updates->tokens().empty());
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
}

TEST(SessionTest, DestructionClosesThroughTheEngineThread) {
  FakeExecutor executor;
  Fixture fixture(executor);
  {
    Session session = open(fixture.runner);
    EXPECT_TRUE(session.valid());
    EXPECT_EQ(executor.open_count(), 1);
  }

  ASSERT_TRUE(wait_for_open_count(executor, 0));
  EXPECT_EQ(executor.closed().size(), 1u);
}

TEST(SessionTest, MoveTransfersOwnershipAndRejectsTheSource) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session original = open(fixture.runner);
  Session owner = std::move(original);

  EXPECT_FALSE(original.valid());
  EXPECT_TRUE(owner.valid());
  EXPECT_EQ(executor.open_count(), 1);

  auto rejected_updates = std::make_shared<Updates>();
  GenerationHandle rejected =
      generate(original, tokens(1), config(1), rejected_updates);
  EXPECT_TRUE(rejected.done());
  EXPECT_EQ(rejected.finish_reason(), FinishReason::Failed);
  ASSERT_TRUE(rejected_updates->wait());
  EXPECT_EQ(rejected_updates->finish(), FinishReason::Failed);
  EXPECT_EQ(rejected_updates->terminal_calls(), 1);
  EXPECT_TRUE(executor.seen().empty());

  auto owner_updates = std::make_shared<Updates>();
  GenerationHandle accepted =
      generate(owner, tokens(1), config(1), owner_updates);
  ASSERT_TRUE(owner_updates->wait());
  EXPECT_EQ(owner_updates->finish(), FinishReason::NewTokenLimit);

  owner = Session{};
  ASSERT_TRUE(wait_for_open_count(executor, 0));
  EXPECT_EQ(executor.closed().size(), 1u);
}

TEST(SessionTest, CapacityRefusalIsReported) {
  FakeExecutor executor;
  executor.capacity = 1;
  Fixture fixture(executor);
  Session first = open(fixture.runner);

  auto refused = fixture.runner.open_session();
  ASSERT_EQ(refused.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(refused.get().has_value());
}

TEST(SessionTest, DestructionCancelsActiveGenerationWithoutWaiting) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle = generate(session, tokens(2), config(1), updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto destroyed = std::async(
      std::launch::async,
      [session = std::move(session)]() mutable { session = Session{}; });
  ASSERT_EQ(destroyed.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(updates->finish().has_value());
  executor.release();

  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
  ASSERT_TRUE(wait_for_open_count(executor, 0));
  EXPECT_EQ(executor.closed().size(), 1u);
}

TEST(GenerationTest, SeedIsSetBeforeTasksExecute) {
  FakeExecutor executor;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();
  GenConfig generation_config = config(1);
  generation_config.seed = 1234;

  Session session = open(fixture.runner);
  const SessionId session_id = last_opened(executor);
  GenerationHandle handle =
      generate(session, tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_TRUE(executor.has_sampling_state(session_id));
  EXPECT_EQ(executor.sampling_seed(session_id), 1234u);
  EXPECT_FALSE(executor.seen().empty());
  EXPECT_FALSE(executor.executed_without_sampling_state());
}

// The policy is installed on the session rather than copied onto every Input,
// so this is the only place it can be observed reaching the executor.
TEST(GenerationTest, SamplingPolicyIsInstalledOnTheSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();
  GenConfig generation_config = config(1);
  generation_config.sampling.temperature = 0.7f;
  generation_config.sampling.top_p = 0.8f;
  generation_config.sampling.top_k = 9;

  Session session = open(fixture.runner);
  const SessionId session_id = last_opened(executor);
  GenerationHandle handle =
      generate(session, tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());

  const std::optional<SamplingParams> installed =
      executor.sampling_params(session_id);
  ASSERT_TRUE(installed.has_value());
  EXPECT_FLOAT_EQ(installed->temperature, 0.7f);
  EXPECT_FLOAT_EQ(installed->top_p, 0.8f);
  EXPECT_EQ(installed->top_k, 9);
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
  const SessionId session_id = last_opened(executor);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(2), config(1), updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_TRUE(executor.has_sampling_state(session_id));
  EXPECT_FALSE(executor.sampling_seed(session_id).has_value());
  EXPECT_FALSE(executor.executed_without_sampling_state());
}

TEST(GenerationTest, ActiveGenerationRejectsDuplicateWithoutReseeding) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  const SessionId session_id = last_opened(executor);

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
  EXPECT_EQ(executor.sampling_seed(session_id), 12u);
  for (const FakeExecutor::Seen& seen : executor.seen()) {
    EXPECT_EQ(seen.sampling_seed, 12u);
  }
}

TEST(GenerationTest, ANewGenerationReseedsTheExistingSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  const SessionId session_id = last_opened(executor);

  auto first_updates = std::make_shared<Updates>();
  GenConfig first_config = config(1);
  first_config.seed = 11;
  GenerationHandle first =
      generate(session, tokens(2), first_config, first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(executor.sampling_seed(session_id), 11u);
  const std::vector<Token> first_tokens = first_updates->tokens();
  ASSERT_EQ(first_tokens.size(), 1u);
  std::mt19937_64 first_rng(11);
  EXPECT_EQ(first_tokens[0], session_id * 1000 + first_rng() % 1000);

  auto second_updates = std::make_shared<Updates>();
  GenConfig second_config = config(1);
  second_config.seed = 22;
  GenerationHandle second =
      generate(session, tokens(1), second_config, second_updates);
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(executor.sampling_seed(session_id), 22u);
  const std::vector<Token> second_tokens = second_updates->tokens();
  ASSERT_EQ(second_tokens.size(), 1u);
  std::mt19937_64 second_rng(22);
  EXPECT_EQ(second_tokens[0], session_id * 1000 + second_rng() % 1000);
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
  // The scheduler's chunk size is what splits the prompt.
  Fixture fixture(executor, 2, 4);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(10), config(1), updates);
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

  const Position after_first = session.position();

  auto second_updates = std::make_shared<Updates>();
  GenerationHandle second =
      generate(session, std::vector<Token>{500}, config(1), second_updates);
  ASSERT_TRUE(second_updates->wait());
  EXPECT_EQ(second_updates->finish(), FinishReason::NewTokenLimit);

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_GE(seen.size(), 3u);
  EXPECT_EQ(seen.back().effective_position(), after_first)
      << "the second turn must append, not restart at 0";
  EXPECT_GT(seen.back().effective_position(), 0);
  EXPECT_EQ(seen.back().size, 2u)
      << "the delta carries the token the first turn emitted but never fed";
}

// Each open_session() gets its own position, so one session's progress does
// not move another.
TEST(DeltaTest, PositionIsPerSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session first = open(fixture.runner);
  Session second = open(fixture.runner);
  const SessionId second_id = last_opened(executor);

  auto first_updates = std::make_shared<Updates>();
  generate(first, tokens(4), config(1), first_updates);
  ASSERT_TRUE(first_updates->wait());
  ASSERT_EQ(first_updates->finish(), FinishReason::NewTokenLimit);

  auto second_updates = std::make_shared<Updates>();
  generate(second, tokens(2), config(1), second_updates);
  ASSERT_TRUE(second_updates->wait());
  ASSERT_EQ(second_updates->finish(), FinishReason::NewTokenLimit);

  for (const FakeExecutor::Seen& slice : executor.seen()) {
    if (slice.session == second_id && slice.size == 2u) {
      EXPECT_EQ(slice.effective_position(), 0)
          << "the second session must open at 0";
    }
  }
}

// The executor no longer names a position, so the continuation lands where the
// runner puts it: right after the delta, since the single produced token is
// not committed until it is fed back.
TEST(DeltaTest, GeneratedTokenFollowsTheDelta) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(2), config(2), updates);
  ASSERT_TRUE(updates->wait());

  const std::vector<FakeExecutor::Seen> seen = executor.seen();
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0].effective_position(), 0);
  EXPECT_EQ(seen[1].effective_position(), 2);
}

// The caller no longer names a position, so the only rejectable input left is
// the delta itself and the token budget.
TEST(DeltaTest, InvalidInitialRangesFailBeforeBeginning) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  const SessionId session_id = last_opened(executor);

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

  EXPECT_FALSE(executor.has_sampling_state(session_id));
  EXPECT_TRUE(executor.seen().empty());
}

// A malformed answer is a contract violation whether or not the runner needed
// the part that is broken. Here the token budget ends the generation on the
// same step that answers badly, and the result is still Failed rather than
// NewTokenLimit.
TEST(DeltaTest, MalformedOutputFailsEvenWhenTheGenerationEnds) {
  FakeExecutor executor;
  executor.empty_tokens = true;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(2), config(1), updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_EQ(executor.seen().size(), 1u) << "nothing further is scheduled";
}

TEST(DeltaTest, MalformedOutputFailsWithoutAnotherTask) {
  for (int malformed = 0; malformed < 2; ++malformed) {
    FakeExecutor executor;
    if (malformed == 0) {
      executor.empty_tokens = true;
    } else {
      executor.wrong_sid = true;
    }
    Fixture fixture(executor);
    Session session = open(fixture.runner);
    auto updates = std::make_shared<Updates>();

    GenerationHandle handle = generate(session, tokens(2), config(2), updates);
    ASSERT_TRUE(updates->wait());
    EXPECT_EQ(updates->finish(), FinishReason::Failed);
    EXPECT_EQ(updates->terminal_calls(), 1);
    EXPECT_EQ(executor.seen().size(), 1u);
  }
}

// A malformed answer means the forward ran but said nothing the runner can
// trust, so the session is retired rather than left at a position built on it.
TEST(FailureTest, MalformedOutputRetiresTheSession) {
  FakeExecutor executor;
  executor.empty_tokens = true;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto first = std::make_shared<Updates>();
  generate(session, tokens(2), config(2), first);
  ASSERT_TRUE(first->wait());
  ASSERT_EQ(first->finish(), FinishReason::Failed);

  auto second = std::make_shared<Updates>();
  generate(session, tokens(1), config(1), second);
  ASSERT_TRUE(second->wait());
  EXPECT_EQ(second->finish(), FinishReason::Failed)
      << "a poisoned session must not accept further generations";
}

// Stopping is the runner's decision alone. The executor has no way to end a
// generation, so all that happens is the runner schedules nothing further.
TEST(DeltaTest, StopTokenSchedulesNothingFurther) {
  FakeExecutor executor;
  executor.stop_token = 999;
  executor.emit_before_stop = 1;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  GenConfig generation_config = config(50);
  generation_config.stop_tokens = {999};
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(session, tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());

  EXPECT_EQ(updates->finish(), FinishReason::StopToken);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_EQ(executor.seen().size(), 2u)
      << "the prompt and one decode, then nothing further is scheduled";
}

TEST(GenerationTest, StopTokenAndBudgetAreAppliedToUpdates) {
  FakeExecutor executor;
  executor.stop_token = 999;
  executor.emit_before_stop = 2;
  Fixture fixture(executor);
  GenConfig generation_config = config(50);
  generation_config.stop_tokens = {999};
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle =
      generate(session, tokens(2), generation_config, updates);
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::StopToken);
  const std::vector<Token> emitted = updates->tokens();
  EXPECT_EQ(emitted.size(), 2u);
  EXPECT_EQ(std::find(emitted.begin(), emitted.end(), 999), emitted.end());
}

// Regression: the runner counts a step's produced tokens as far as the
// executor committed them, so the next turn resumes past them rather than on
// top of them. The last token of a run is not committed until it is fed back,
// which is what `pending` carries across the turn boundary.
TEST(SpeculativeTest, TerminalStepPositionCarriesToTheNextTurn) {
  FakeExecutor executor;
  executor.tokens_per_decode = 3;
  Fixture fixture(executor);
  Session session = open(fixture.runner);

  auto first = std::make_shared<Updates>();
  generate(session, tokens(2), config(4), first);
  ASSERT_TRUE(first->wait());
  ASSERT_EQ(first->finish(), FinishReason::NewTokenLimit);
  const std::size_t emitted = first->tokens().size();
  const Position after_first = session.position();

  // The prompt plus every token the callback delivered, less the last one,
  // which is emitted but not fed until the next turn carries it.
  EXPECT_EQ(after_first, static_cast<Position>(2 + emitted - 1));

  auto second = std::make_shared<Updates>();
  generate(session, tokens(1), config(1), second);
  ASSERT_TRUE(second->wait());
  ASSERT_EQ(second->finish(), FinishReason::NewTokenLimit);

  EXPECT_EQ(executor.seen().back().effective_position(), after_first)
      << "the second turn must resume at the carried token, not past it";
  EXPECT_EQ(executor.seen().back().size, 2u)
      << "the delta carries the token the first turn emitted but never fed";
}

TEST(SpeculativeTest, AcceptedRunUsesExecutorPosition) {
  FakeExecutor executor;
  executor.tokens_per_decode = 3;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(2), config(7), updates);
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
  Runner runner(executor, scheduler);
  Session session = open(runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(2), config(2), updates);
  ASSERT_TRUE(updates->wait());
  handle.wait();
  EXPECT_EQ(updates->finish(), FinishReason::Failed);
  EXPECT_EQ(updates->terminal_calls(), 1);
  EXPECT_TRUE(executor.seen().empty());
}

TEST(FailureTest, FailureOnLeadingPrefillChunksEndsGeneration) {
  FakeExecutor executor;
  executor.fail_batches_from = 0;
  Fixture fixture(executor, 1, 4);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();

  GenerationHandle handle = generate(session, tokens(10), config(1), updates);

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
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(2), config(100000), updates);

  handle.cancel();
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
}

// The callback runs on the engine thread while handle_output_ still holds a
// reference into sessions_. Retiring the session from inside it must be safe:
// the close is queued, not applied, so the record survives the call and the
// generation ends Cancelled on the next pass.
TEST(CancelTest, CallbackMayRetireItsOwnSession) {
  FakeExecutor executor;
  Fixture fixture(executor);
  auto updates = std::make_shared<Updates>();
  std::optional<Session> session = open(fixture.runner);
  bool dropped = false;

  GenerationHandle handle = session->generate_async(
      tokens(2),
      config(100000),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        (*updates)(emitted, finish);
        if (!finish && !dropped) {
          dropped = true;
          session.reset();
        }
      });

  ASSERT_TRUE(updates->wait());
  EXPECT_TRUE(dropped) << "the test never exercised the reentrant drop";
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
  EXPECT_EQ(updates->terminal_calls(), 1);
}

// The session tracks what the executor consumed, not what the runner asked
// for, so a cancelled generation leaves it where the executor actually stopped
// rather than back where its first chunk started.
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

// Cancelling during decode leaves the session holding exactly the delta plus
// the tokens the callback delivered: every position the runner advanced was
// paid for by a token the caller was told about.
//
// The cancel lands while the executor is held, so the continuation carrying the
// last delivered token is dropped before it can run. That is the case that
// loses a token unless something carries it across the turn.
TEST(CancelTest, DecodeCancelAdvancesOnlyForDeliveredTokens) {
  FakeExecutor executor;
  executor.tokens_per_decode = 3;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  ASSERT_EQ(session.position(), 0);

  const std::vector<Token> delta = tokens(4);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle = generate(session, delta, config(100000), updates);

  // Let the prompt finish prefilling and a run land, so the cancel falls during
  // decode rather than part way through the delta.
  while (updates->tokens().empty()) {
    std::this_thread::yield();
  }
  // Freeze the engine inside execute, then cancel, so the queued continuation
  // is dropped rather than executed.
  executor.hold();
  handle.cancel();
  executor.release();
  ASSERT_TRUE(updates->wait());
  ASSERT_EQ(updates->finish(), FinishReason::Cancelled);

  const std::size_t delivered = updates->tokens().size();
  ASSERT_GT(delivered, 0u);

  // Never more than the delta plus what the caller received. It sits one short
  // while a delivered token is still waiting to be fed.
  const auto grown = static_cast<std::size_t>(session.position());
  EXPECT_LE(grown, delta.size() + delivered);
  EXPECT_GE(grown, delta.size() + delivered - 1);

  // The shortfall is a carry, not a loss: the next turn feeds it ahead of its
  // own delta, so the executor sees every delivered token.
  const std::size_t carried = delta.size() + delivered - grown;
  auto resumed = std::make_shared<Updates>();
  generate(session, tokens(1), config(1), resumed);
  ASSERT_TRUE(resumed->wait());
  ASSERT_EQ(resumed->finish(), FinishReason::NewTokenLimit);

  const FakeExecutor::Seen last_delta = executor.seen().back();
  EXPECT_EQ(last_delta.effective_position(), static_cast<Position>(grown));
  EXPECT_EQ(last_delta.size, 1u + carried)
      << "a token delivered but not yet fed must ride with the next delta";
}

TEST(SessionTest, DestroyingASessionEndsItsGeneration) {
  FakeExecutor executor;
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  auto updates = std::make_shared<Updates>();
  GenerationHandle handle =
      generate(session, tokens(2), config(100000), updates);

  session = Session{};
  ASSERT_TRUE(updates->wait());
  EXPECT_EQ(updates->finish(), FinishReason::Cancelled);
}

TEST(ShutdownTest, EndsLiveAndRejectsNewWork) {
  FakeExecutor executor;
  executor.hold();
  Fixture fixture(executor);
  Session session = open(fixture.runner);
  const std::vector<SessionId> opened = executor.opened();
  auto live_updates = std::make_shared<Updates>();
  std::atomic<bool> invalid_in_terminal_callback{false};
  GenerationHandle live = session.generate_async(
      tokens(2),
      config(100000),
      [&](const std::vector<Token>& emitted,
          std::optional<FinishReason> finish) {
        if (finish) {
          invalid_in_terminal_callback.store(!session.valid());
        }
        (*live_updates)(emitted, finish);
      });
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto stopping =
      std::async(std::launch::async, [&] { fixture.runner.shutdown(); });
  std::vector<std::future<std::optional<Session>>> admitted_opens;
  while (true) {
    auto opened_session = fixture.runner.open_session();
    if (opened_session.wait_for(std::chrono::milliseconds(1)) ==
        std::future_status::ready) {
      EXPECT_FALSE(opened_session.get().has_value());
      break;
    }
    admitted_opens.push_back(std::move(opened_session));
  }
  executor.release();
  ASSERT_EQ(stopping.wait_for(kTimeout), std::future_status::ready);
  for (auto& opened_session : admitted_opens) {
    ASSERT_EQ(opened_session.wait_for(kTimeout), std::future_status::ready);
    EXPECT_FALSE(opened_session.get().has_value());
  }
  ASSERT_TRUE(live_updates->wait());
  EXPECT_EQ(live_updates->finish(), FinishReason::Cancelled);
  EXPECT_TRUE(invalid_in_terminal_callback.load());
  EXPECT_FALSE(session.valid());
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed(), opened);

  auto rejected_open = fixture.runner.open_session();
  ASSERT_EQ(rejected_open.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(rejected_open.get().has_value());

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
  GenerationHandle handle = generate(session, tokens(2), config(1), updates);
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
  GenerationHandle handle = generate(session, tokens(2), config(2), updates);
  while (!executor.in_execute()) {
    std::this_thread::yield();
  }

  auto owner =
      std::async(std::launch::async, [&] { fixture.runner.shutdown(); });
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
  EXPECT_EQ(
      owner.wait_for(std::chrono::milliseconds(20)),
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
  EXPECT_EQ(executor.closed(), executor.opened());
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
  std::vector<SessionId> opened = executor.opened();
  second = Session{};
  ASSERT_TRUE(wait_for_open_count(executor, 2));

  fixture.runner.shutdown();

  EXPECT_EQ(executor.open_count(), 0);
  std::vector<SessionId> closed = executor.closed();
  std::sort(opened.begin(), opened.end());
  std::sort(closed.begin(), closed.end());
  EXPECT_EQ(closed, opened);
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

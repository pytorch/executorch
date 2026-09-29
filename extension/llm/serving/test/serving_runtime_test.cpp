/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/serving_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::DecodeFirstScheduler;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::testing::FakeExecutor;
using executorch::extension::llm::serving::ErrorCode;
using executorch::extension::llm::serving::LifecycleResult;
using executorch::extension::llm::serving::ServingError;
using executorch::extension::llm::serving::ServingRuntime;
using executorch::extension::llm::serving::ServingRuntimeConfig;

namespace {

constexpr std::chrono::seconds kTimeout{5};

static_assert(!std::is_copy_constructible<ServingRuntime>::value);
static_assert(!std::is_move_constructible<ServingRuntime>::value);

class Gate {
 public:
  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
  }

  void arrive() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrivals_;
    cv_.notify_all();
    cv_.wait(lock, [this] { return !held_; });
  }

  bool wait_for(std::size_t arrivals) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, kTimeout, [&] { return arrivals_ >= arrivals; });
  }

  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      held_ = false;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t arrivals_ = 0;
  bool held_ = false;
};

class LifecycleExecutor : public FakeExecutor {
 public:
  bool initialize() override {
    note_thread();
    initializing.arrive();
#if ET_HAS_EXCEPTIONS
    if (throw_initialize) {
      throw std::runtime_error("initialization failed");
    }
#endif
    return FakeExecutor::initialize();
  }

  std::optional<SessionId> open_session() override {
    note_thread();
    opening.arrive();
    ++open_attempts;
    if (refuse_next_open.exchange(false)) {
      return std::nullopt;
    }
    auto session = FakeExecutor::open_session();
    opened_session.arrive();
    return session;
  }

  void close_session(SessionId session) override {
    note_thread();
    closing.arrive();
    FakeExecutor::close_session(session);
  }

  std::optional<SessionId> clone(SessionId, Position) override {
    ++clone_calls;
    return std::nullopt;
  }

  void release_all() {
    initializing.release();
    opening.release();
    opened_session.release();
    closing.release();
  }

  std::vector<std::thread::id> threads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return threads_;
  }

  Gate initializing;
  Gate opening;
  Gate opened_session;
  Gate closing;
  bool throw_initialize = false;
  std::atomic<bool> refuse_next_open{false};
  std::atomic<int> open_attempts{0};
  std::atomic<int> clone_calls{0};

 private:
  void note_thread() {
    std::lock_guard<std::mutex> lock(mutex_);
    threads_.push_back(std::this_thread::get_id());
  }

  mutable std::mutex mutex_;
  std::vector<std::thread::id> threads_;
};

class ServingRuntimeTest : public ::testing::Test {
 protected:
  void start(ServingRuntimeConfig config = {2, 128, 16}) {
    runtime = std::make_unique<ServingRuntime>(
        executor, DecodeFirstScheduler::create(), config);
  }

  void TearDown() override {
    // Release gates even after a fatal assertion, before the runtime joins.
    executor.release_all();
    runtime.reset();
    EXPECT_EQ(executor.clone_calls.load(), 0);
    EXPECT_TRUE(executor.seen().empty());
  }

  LifecycleExecutor executor;
  std::unique_ptr<ServingRuntime> runtime;
};

struct JoiningThreads {
  explicit JoiningThreads(LifecycleExecutor& executor) : executor(executor) {}

  ~JoiningThreads() {
    executor.release_all();
    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  LifecycleExecutor& executor;
  std::vector<std::thread> threads;
};

LifecycleResult result(std::future<LifecycleResult> future) {
  if (future.wait_for(kTimeout) != std::future_status::ready) {
    ADD_FAILURE() << "lifecycle operation did not settle";
    return ServingError{ErrorCode::Internal, "test timeout"};
  }
  return future.get();
}

void expect_error(std::future<LifecycleResult> future, ErrorCode code) {
  auto error = result(std::move(future));
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, code);
  EXPECT_FALSE(error->message.empty());
}

template <class Predicate>
bool wait_until(Predicate predicate) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

TEST_F(ServingRuntimeTest, OpenQueuesDuringInitializationAndReportsReadiness) {
  executor.initializing.hold();
  executor.opening.hold();
  start({1, 4096, 4});
  ASSERT_TRUE(executor.initializing.wait_for(1));
  const auto info = runtime->info();
  EXPECT_FALSE(info.ready);
  EXPECT_EQ(info.max_sessions, 1u);
  EXPECT_EQ(info.max_context_length, 4096u);
  EXPECT_EQ(info.active_sessions, 0u);

  auto opened = runtime->open_session_async("session");
  EXPECT_EQ(
      opened.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  executor.initializing.release();
  ASSERT_TRUE(executor.opening.wait_for(1));
  EXPECT_TRUE(runtime->info().ready);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(
      opened.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  executor.opening.release();
  EXPECT_FALSE(result(std::move(opened)));
  EXPECT_TRUE(runtime->info().ready); // Full capacity is still initialized.
  EXPECT_EQ(executor.initialize_calls(), 1);
  EXPECT_FALSE(executor.called_before_initialize());
}

TEST_F(ServingRuntimeTest, DuplicateOpenAndCloseAreIdempotent) {
  executor.opening.hold();
  start();
  auto first = runtime->open_session_async("session");
  ASSERT_TRUE(executor.opening.wait_for(1));
  auto duplicate = runtime->open_session_async("session");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(first)));
  EXPECT_FALSE(result(std::move(duplicate)));
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  EXPECT_EQ(executor.open_attempts.load(), 1);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  runtime->shutdown();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, SameKeyOperationsPreserveAdmissionOrder) {
  executor.opening.hold();
  executor.capacity = 1;
  start({1, 128, 8});
  auto opened = runtime->open_session_async("session");
  ASSERT_TRUE(executor.opening.wait_for(1));
  auto reset = runtime->reset_session_async("session");
  auto closed = runtime->close_session_async("session");
  auto reopened = runtime->open_session_async("session");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(opened)));
  EXPECT_FALSE(result(std::move(reset)));
  EXPECT_FALSE(result(std::move(closed)));
  EXPECT_FALSE(result(std::move(reopened)));
  EXPECT_EQ(executor.opened().size(), 3u);
  EXPECT_EQ(executor.closed().size(), 2u);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
}

TEST_F(ServingRuntimeTest, OpenCannotBypassPendingResetAndClose) {
  start();
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  executor.opening.hold();
  auto reset = runtime->reset_session_async("session");
  ASSERT_TRUE(executor.opening.wait_for(2));
  auto closed = runtime->close_session_async("session");
  auto opened = runtime->open_session_async("session");
  EXPECT_EQ(
      opened.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  executor.opening.release();
  EXPECT_FALSE(result(std::move(reset)));
  EXPECT_FALSE(result(std::move(closed)));
  EXPECT_FALSE(result(std::move(opened)));
  EXPECT_EQ(executor.opened().size(), 3u);
}

TEST_F(ServingRuntimeTest, CloseAcknowledgesLogicalNotPhysicalCleanup) {
  start();
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  executor.closing.hold();
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  ASSERT_TRUE(executor.closing.wait_for(1));
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.open_count(), 1);
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  executor.closing.release();
  runtime->shutdown();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, LogicalCapacityRejectsWithoutCallingExecutor) {
  executor.opening.hold();
  start({1, 128, 4});
  auto first = runtime->open_session_async("first");
  ASSERT_TRUE(executor.opening.wait_for(1));
  auto second = runtime->open_session_async("second");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(first)));
  expect_error(std::move(second), ErrorCode::CapacityExceeded);
  EXPECT_EQ(executor.open_attempts.load(), 1);
  EXPECT_FALSE(result(runtime->close_session_async("first")));
  EXPECT_FALSE(result(runtime->open_session_async("second")));
  EXPECT_EQ(runtime->info().active_sessions, 1u);
}

TEST_F(ServingRuntimeTest, ExecutorRefusalReleasesInitialReservation) {
  executor.capacity = 1;
  start({2, 0, 4});
  EXPECT_FALSE(result(runtime->open_session_async("first")));
  expect_error(
      runtime->open_session_async("second"), ErrorCode::CapacityExceeded);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_FALSE(result(runtime->close_session_async("first")));
  EXPECT_FALSE(result(runtime->open_session_async("second")));
}

TEST_F(ServingRuntimeTest, ResetRetainsSlotUntilColdReplacementCompletes) {
  executor.capacity = 1;
  start({1, 128, 4});
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  const auto original = executor.opened().front();
  executor.closing.hold();
  executor.opening.hold();
  auto reset = runtime->reset_session_async("session");
  ASSERT_TRUE(executor.closing.wait_for(1));
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(
      reset.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  executor.closing.release();
  ASSERT_TRUE(executor.opening.wait_for(2));
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed(), (std::vector<SessionId>{original}));
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  auto other = runtime->open_session_async("other");
  EXPECT_EQ(
      reset.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  executor.opening.release();
  EXPECT_FALSE(result(std::move(reset)));
  expect_error(std::move(other), ErrorCode::CapacityExceeded);
  ASSERT_EQ(executor.opened().size(), 2u);
  EXPECT_NE(executor.opened().back(), original);
  EXPECT_EQ(executor.open_count(), 1);
}

TEST_F(ServingRuntimeTest, FailedResetRetainsUnavailableSlotAndCanRetry) {
  start({1, 128, 4});
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  executor.refuse_next_open = true;
  auto failed = result(runtime->reset_session_async("session"));
  ASSERT_TRUE(failed);
  EXPECT_EQ(failed->code, ErrorCode::CapacityExceeded);
  EXPECT_NE(failed->message.find("old state is gone"), std::string::npos);
  EXPECT_NE(failed->message.find("unavailable"), std::string::npos);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_TRUE(runtime->info().ready);
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.closed().size(), 1u);
  expect_error(runtime->open_session_async("session"), ErrorCode::NotReady);
  expect_error(
      runtime->open_session_async("other"), ErrorCode::CapacityExceeded);
  EXPECT_EQ(executor.open_attempts.load(), 2);
  EXPECT_FALSE(result(runtime->reset_session_async("session")));
  EXPECT_EQ(executor.open_attempts.load(), 3);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  EXPECT_EQ(executor.open_attempts.load(), 3);
}

TEST_F(ServingRuntimeTest, CloseReleasesUnavailableSlot) {
  start({1, 128, 4});
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  executor.refuse_next_open = true;
  expect_error(
      runtime->reset_session_async("session"), ErrorCode::CapacityExceeded);
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_FALSE(result(runtime->open_session_async("other")));
  runtime->shutdown();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, QueueBoundIncludesExecutingOperation) {
  executor.opening.hold();
  start({2, 128, 2});
  auto first = runtime->open_session_async("first");
  ASSERT_TRUE(executor.opening.wait_for(1));
  auto second = runtime->open_session_async("second");
  auto rejected = runtime->close_session_async("first");
  EXPECT_EQ(
      rejected.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  expect_error(std::move(rejected), ErrorCode::CapacityExceeded);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  executor.opening.release();
  EXPECT_FALSE(result(std::move(first)));
  EXPECT_FALSE(result(std::move(second)));
  EXPECT_EQ(runtime->info().active_sessions, 2u);
  EXPECT_FALSE(result(runtime->close_session_async("first")));
}

TEST_F(ServingRuntimeTest, InitializationFailureSettlesQueuedAndNewOperations) {
  executor.fail_initialize = true;
  executor.initializing.hold();
  start();
  ASSERT_TRUE(executor.initializing.wait_for(1));
  auto first = runtime->open_session_async("first");
  auto second = runtime->open_session_async("second");
  executor.initializing.release();
  expect_error(std::move(first), ErrorCode::NotReady);
  expect_error(std::move(second), ErrorCode::NotReady);
  expect_error(runtime->open_session_async("third"), ErrorCode::NotReady);
  EXPECT_FALSE(runtime->info().ready);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.open_attempts.load(), 0);
  runtime->shutdown();
}

#if ET_HAS_EXCEPTIONS
TEST_F(ServingRuntimeTest, InitializationExceptionBecomesNotReady) {
  executor.throw_initialize = true;
  executor.initializing.hold();
  start();
  ASSERT_TRUE(executor.initializing.wait_for(1));
  auto opened = runtime->open_session_async("session");
  executor.initializing.release();
  expect_error(std::move(opened), ErrorCode::NotReady);
  expect_error(runtime->open_session_async("another"), ErrorCode::NotReady);
  EXPECT_FALSE(runtime->info().ready);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.open_attempts.load(), 0);
  runtime->shutdown();
}
#endif

TEST_F(ServingRuntimeTest, ShutdownDiscardsLateOpenAndSettlesFullQueue) {
  executor.opened_session.hold();
  start({2, 128, 3});
  auto opened = runtime->open_session_async("session");
  ASSERT_TRUE(executor.opened_session.wait_for(1));
  auto reset = runtime->reset_session_async("session");
  auto closed = runtime->close_session_async("session");
  EXPECT_EQ(executor.open_count(), 1);
  EXPECT_TRUE(runtime->info().ready);
  std::atomic<int> completed{0};
  JoiningThreads callers(executor);
  for (int i = 0; i < 4; ++i) {
    callers.threads.emplace_back([&] {
      runtime->shutdown();
      ++completed;
    });
  }
  ASSERT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  EXPECT_EQ(completed.load(), 0);
  expect_error(runtime->open_session_async("new"), ErrorCode::NotReady);
  executor.opened_session.release();
  ASSERT_TRUE(wait_until([&] { return completed.load() == 4; }));
  expect_error(std::move(opened), ErrorCode::NotReady);
  expect_error(std::move(reset), ErrorCode::NotReady);
  expect_error(std::move(closed), ErrorCode::NotReady);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.closed(), executor.opened());
  runtime->shutdown();
  expect_error(runtime->close_session_async("session"), ErrorCode::NotReady);
  expect_error(runtime->reset_session_async("session"), ErrorCode::NotReady);
}

TEST_F(ServingRuntimeTest, ShutdownDuringResetClosesOldAndLateReplacementOnce) {
  start();
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  executor.opened_session.hold();
  auto reset = runtime->reset_session_async("session");
  ASSERT_TRUE(executor.opened_session.wait_for(2));
  EXPECT_EQ(executor.opened().size(), 2u);
  EXPECT_EQ(executor.closed().size(), 1u);
  JoiningThreads callers(executor);
  callers.threads.emplace_back([&] { runtime->shutdown(); });
  ASSERT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  executor.opened_session.release();
  callers.threads.front().join();
  expect_error(std::move(reset), ErrorCode::NotReady);
  EXPECT_EQ(executor.closed(), executor.opened());
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
}

TEST_F(ServingRuntimeTest, ConcurrentAdmissionOpensOneSession) {
  constexpr int kCallers = 8;
  executor.opening.hold();
  start({1, 128, kCallers});
  std::vector<std::future<LifecycleResult>> futures(kCallers);
  std::atomic<int> admitted{0};
  JoiningThreads callers(executor);
  for (int i = 0; i < kCallers; ++i) {
    callers.threads.emplace_back([&, i] {
      futures[i] = runtime->open_session_async("session");
      ++admitted;
    });
  }
  ASSERT_TRUE(wait_until([&] { return admitted.load() == kCallers; }));
  ASSERT_TRUE(executor.opening.wait_for(1));
  executor.opening.release();
  for (auto& future : futures) {
    EXPECT_FALSE(result(std::move(future)));
  }
  EXPECT_EQ(executor.open_attempts.load(), 1);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
}

TEST_F(ServingRuntimeTest, DestructionJoinsAndClosesAllOwnedSessionsOnce) {
  start();
  EXPECT_FALSE(result(runtime->open_session_async("first")));
  EXPECT_FALSE(result(runtime->open_session_async("second")));
  EXPECT_FALSE(result(runtime->reset_session_async("first")));
  runtime.reset();
  EXPECT_EQ(executor.open_count(), 0);
  auto opened = executor.opened();
  auto closed = executor.closed();
  std::sort(opened.begin(), opened.end());
  std::sort(closed.begin(), closed.end());
  EXPECT_EQ(opened, closed);
  const auto threads = executor.threads();
  ASSERT_FALSE(threads.empty());
  EXPECT_NE(threads.front(), std::this_thread::get_id());
  EXPECT_TRUE(std::all_of(threads.begin(), threads.end(), [&](auto id) {
    return id == threads.front();
  }));
}

TEST_F(ServingRuntimeTest, AbandonedFutureDoesNotAbandonSessionOwnership) {
  executor.opening.hold();
  start();
  {
    auto abandoned = runtime->open_session_async("session");
    ASSERT_TRUE(executor.opening.wait_for(1));
  }
  auto duplicate = runtime->open_session_async("session");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(duplicate)));
  EXPECT_EQ(executor.open_attempts.load(), 1);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  runtime.reset();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, InvalidKeysAndMissingResetDoNotAllocateSlots) {
  start();
  expect_error(runtime->open_session_async(""), ErrorCode::InvalidArgument);
  expect_error(runtime->close_session_async(""), ErrorCode::InvalidArgument);
  expect_error(runtime->reset_session_async(""), ErrorCode::InvalidArgument);
  expect_error(
      runtime->reset_session_async("missing"), ErrorCode::SessionNotFound);
  EXPECT_FALSE(result(runtime->close_session_async("missing")));
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.open_attempts.load(), 0);
}

TEST_F(ServingRuntimeTest, InvalidConfigurationIsInertWithoutExceptions) {
  for (const auto config :
       {ServingRuntimeConfig{0, 128, 4}, ServingRuntimeConfig{1, 128, 0}}) {
    start(config);
    EXPECT_FALSE(runtime->info().ready);
    expect_error(
        runtime->open_session_async("session"), ErrorCode::InvalidArgument);
    expect_error(
        runtime->reset_session_async("session"), ErrorCode::InvalidArgument);
    expect_error(
        runtime->close_session_async("session"), ErrorCode::InvalidArgument);
    runtime->shutdown();
    runtime.reset();
  }
  runtime = std::make_unique<ServingRuntime>(
      executor, nullptr, ServingRuntimeConfig{});
  EXPECT_FALSE(runtime->info().ready);
  expect_error(
      runtime->open_session_async("session"), ErrorCode::InvalidArgument);
  EXPECT_EQ(executor.initialize_calls(), 0);
}

} // namespace

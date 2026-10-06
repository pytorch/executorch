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
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using executorch::extension::llm::batching::DecodeFirstScheduler;
using executorch::extension::llm::batching::Position;
using executorch::extension::llm::batching::SessionId;
using executorch::extension::llm::batching::testing::FakeExecutor;
using executorch::extension::llm::serving::ErrorCode;
using executorch::extension::llm::serving::LifecycleCallback;
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

  bool wait_for(std::size_t arrivals, std::chrono::seconds timeout = kTimeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return arrivals_ >= arrivals; });
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
  void SetUp() override {
    executorch::runtime::runtime_init();
  }

  void start(ServingRuntimeConfig config = {2, 128, 16}) {
    runtime = std::make_unique<ServingRuntime>(
        executor, DecodeFirstScheduler::create(), config);
  }

  void watch_callbacks() {
    // Reentry regressions can deadlock inside info()/submit() or teardown.
    callback_watchdog = std::thread([this] {
      if (!test_finished.wait_for(1, kTimeout * 6)) {
        ADD_FAILURE() << "callback test or teardown deadlocked";
        std::abort();
      }
    });
  }

  void TearDown() override {
    // Release gates even after a fatal assertion, before the runtime joins.
    callback_gate.release();
    cleanup_gate.release();
    executor.release_all();
    for (auto& caller : callback_callers) {
      if (caller.joinable()) {
        caller.join();
      }
    }
    if (runtime) {
      runtime->shutdown();
    }
    runtime.reset();
    EXPECT_EQ(executor.clone_calls.load(), 0);
    EXPECT_TRUE(executor.seen().empty());
    test_finished.arrive();
    if (callback_watchdog.joinable()) {
      callback_watchdog.join();
    }
  }

  LifecycleExecutor executor;
  std::unique_ptr<ServingRuntime> runtime;
  Gate callback_gate;
  Gate cleanup_gate;
  std::vector<std::thread> callback_callers;
  Gate test_finished;
  std::thread callback_watchdog;
  struct SmallCallbackState {
    std::atomic<int> destructions{0};
    std::atomic<int> calls{0};
    std::thread::id thread;
  } small_callback_state;
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

TEST_F(ServingRuntimeTest, CallbackAndFutureOperationsPreserveAdmissionOrder) {
  watch_callbacks();
  executor.opening.hold();
  callback_gate.hold();
  start({1, 128, 4});
  const auto caller = std::this_thread::get_id();
  auto calls = std::make_shared<std::atomic<int>>(0);
  runtime->open_session_async("session", [this, calls, caller](auto error) {
    EXPECT_FALSE(error);
    EXPECT_EQ(calls->fetch_add(1), 0);
    const auto control = std::this_thread::get_id();
    EXPECT_NE(control, caller);
    for (auto engine : executor.threads()) {
      EXPECT_NE(control, engine);
    }
    EXPECT_EQ(runtime->info().active_sessions, 1u);
    callback_gate.arrive();
  });
  ASSERT_TRUE(executor.opening.wait_for(1));
  auto reset = runtime->reset_session_async("session");
  runtime->close_session_async("session", [this, calls, caller](auto error) {
    EXPECT_FALSE(error);
    EXPECT_EQ(calls->fetch_add(1), 1);
    EXPECT_NE(std::this_thread::get_id(), caller);
    for (auto engine : executor.threads()) {
      EXPECT_NE(std::this_thread::get_id(), engine);
    }
    EXPECT_EQ(runtime->info().active_sessions, 0u);
  });
  auto reopened = runtime->open_session_async("session");
  executor.opening.release();
  ASSERT_TRUE(callback_gate.wait_for(1));
  EXPECT_EQ(
      reset.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(executor.open_attempts.load(), 1);
  callback_gate.release();
  EXPECT_FALSE(result(std::move(reset)));
  EXPECT_FALSE(result(std::move(reopened)));
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(executor.opened().size(), 3u);
  runtime->shutdown();
  EXPECT_EQ(calls->load(), 2);
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, CallbackCanSubmitFollowupAfterReleasingOnlyPermit) {
  watch_callbacks();
  start({1, 128, 1});
  const auto caller = std::this_thread::get_id();
  auto calls = std::make_shared<std::atomic<int>>(0);
  runtime->open_session_async("session", [this, calls, caller](auto error) {
    EXPECT_FALSE(error);
    EXPECT_EQ(calls->fetch_add(1), 0);
    const auto control = std::this_thread::get_id();
    EXPECT_NE(control, caller);
    for (auto engine : executor.threads()) {
      EXPECT_NE(control, engine);
    }
    EXPECT_EQ(runtime->info().active_sessions, 1u);
    runtime->reset_session_async("session", [this, calls, control](auto reset) {
      EXPECT_FALSE(reset);
      EXPECT_EQ(calls->fetch_add(1), 1);
      EXPECT_EQ(std::this_thread::get_id(), control);
      EXPECT_EQ(runtime->info().active_sessions, 1u);
      runtime->close_session_async(
          "session", [this, calls, control](auto closed) {
            EXPECT_FALSE(closed);
            EXPECT_EQ(calls->fetch_add(1), 2);
            EXPECT_EQ(std::this_thread::get_id(), control);
            EXPECT_EQ(runtime->info().active_sessions, 0u);
            callback_gate.arrive();
          });
    });
  });
  ASSERT_TRUE(callback_gate.wait_for(1));
  runtime->shutdown();
  EXPECT_EQ(calls->load(), 3);
  EXPECT_EQ(executor.open_attempts.load(), 2);
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, CallbackRejectionsAreInlineAndAllowReentry) {
  watch_callbacks();
  using Operation = void (ServingRuntime::*)(std::string, LifecycleCallback);
  const Operation operations[] = {
      &ServingRuntime::open_session_async,
      &ServingRuntime::reset_session_async,
      &ServingRuntime::close_session_async};
  enum class Rejection { Validation, Inert, Stopped, Saturated };
  for (auto rejection :
       {Rejection::Validation,
        Rejection::Inert,
        Rejection::Stopped,
        Rejection::Saturated}) {
    SCOPED_TRACE(static_cast<int>(rejection));
    executor.opening.hold();
    start({rejection == Rejection::Inert ? 0u : 1u, 128, 1});
    if (rejection == Rejection::Stopped) {
      runtime->shutdown();
    } else if (rejection == Rejection::Saturated) {
      runtime->open_session_async("occupied", {});
      ASSERT_TRUE(executor.opening.wait_for(1));
    }
    const std::string key = rejection == Rejection::Validation ? "" : "session";
    const auto expected = rejection == Rejection::Stopped ? ErrorCode::NotReady
        : rejection == Rejection::Saturated ? ErrorCode::CapacityExceeded
                                            : ErrorCode::InvalidArgument;
    const auto caller = std::this_thread::get_id();
    auto calls = std::make_shared<std::atomic<int>>(0);
    for (auto operation : operations) {
      const auto before = calls->load();
      (runtime.get()->*operation)(
          key, [this, calls, caller, expected, key](auto error) {
            EXPECT_EQ(std::this_thread::get_id(), caller);
            ASSERT_TRUE(error);
            EXPECT_EQ(error->code, expected);
            EXPECT_FALSE(error->message.empty());
            (void)runtime->info();
            ++*calls;
            runtime->close_session_async(
                key, [calls, caller, expected](auto nested) {
                  EXPECT_EQ(std::this_thread::get_id(), caller);
                  ASSERT_TRUE(nested);
                  EXPECT_EQ(nested->code, expected);
                  ++*calls;
                });
          });
      EXPECT_EQ(calls->load(), before + 2);
    }
    executor.opening.release();
    runtime->shutdown();
    EXPECT_EQ(calls->load(), 6);
    runtime.reset();
  }
}

TEST_F(ServingRuntimeTest, QueuedCallbacksSettleOnceOnInitFailureAndShutdown) {
  watch_callbacks();
  for (bool shutdown : {false, true}) {
    SCOPED_TRACE(shutdown);
    executor.fail_initialize = !shutdown;
    executor.initializing.hold();
    executor.opened_session.hold();
    const auto previous_threads = executor.threads().size();
    start({3, 128, 3});
    struct State {
      std::atomic<int> calls[3]{};
      std::mutex mutex;
      std::thread::id control;
    };
    auto state = std::make_shared<State>();
    const auto caller = std::this_thread::get_id();
    auto completion = [this, state, caller, previous_threads](
                          int index, LifecycleResult error) {
      EXPECT_EQ(state->calls[index].fetch_add(1), 0);
      EXPECT_NE(std::this_thread::get_id(), caller);
      const auto engines = executor.threads();
      for (auto i = previous_threads; i < engines.size(); ++i) {
        EXPECT_NE(std::this_thread::get_id(), engines[i]);
      }
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->control == std::thread::id{}) {
          state->control = std::this_thread::get_id();
        }
        EXPECT_EQ(std::this_thread::get_id(), state->control);
      }
      EXPECT_FALSE(runtime->info().ready);
      ASSERT_TRUE(error);
      EXPECT_EQ(error->code, ErrorCode::NotReady);
    };
    runtime->open_session_async(
        "first", [completion](auto error) { completion(0, error); });
    if (shutdown) {
      executor.initializing.release();
      ASSERT_TRUE(executor.opened_session.wait_for(1));
      runtime->reset_session_async(
          "first", [completion](auto error) { completion(1, error); });
      runtime->close_session_async(
          "first", [completion](auto error) { completion(2, error); });
    } else {
      runtime->open_session_async(
          "second", [completion](auto error) { completion(1, error); });
      runtime->open_session_async(
          "third", [completion](auto error) { completion(2, error); });
    }
    expect_error(
        runtime->open_session_async("overflow"), ErrorCode::CapacityExceeded);
    if (shutdown) {
      callback_callers.emplace_back([this] { runtime->shutdown(); });
      ASSERT_TRUE(wait_until([this] { return !runtime->info().ready; }));
    }
    executor.release_all();
    ASSERT_TRUE(wait_until([state] {
      return state->calls[0].load() == 1 && state->calls[1].load() == 1 &&
          state->calls[2].load() == 1;
    }));
    runtime->shutdown();
    for (auto& thread : callback_callers) {
      thread.join();
    }
    callback_callers.clear();
    for (auto& count : state->calls) {
      EXPECT_EQ(count.load(), 1);
    }
    EXPECT_EQ(runtime->info().active_sessions, 0u);
    EXPECT_EQ(executor.closed(), executor.opened());
    runtime.reset();
  }
}

#if ET_HAS_EXCEPTIONS
TEST_F(ServingRuntimeTest, ThrowingCallbacksDoNotRollbackOrStopControl) {
  watch_callbacks();
  start({1, 128, 1});
  auto calls = std::make_shared<std::atomic<int>>(0);
  runtime->open_session_async("session", [this, calls](auto error) {
    EXPECT_FALSE(error);
    EXPECT_EQ(runtime->info().active_sessions, 1u);
    ++*calls;
    callback_gate.arrive();
    throw std::runtime_error("admitted callback failed");
  });
  ASSERT_TRUE(callback_gate.wait_for(1));
  EXPECT_FALSE(result(runtime->open_session_async("session")));
  EXPECT_EQ(executor.open_attempts.load(), 1);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  const auto caller = std::this_thread::get_id();
  EXPECT_NO_THROW(
      runtime->reset_session_async("", [this, calls, caller](auto error) {
        ASSERT_TRUE(error);
        EXPECT_EQ(error->code, ErrorCode::InvalidArgument);
        EXPECT_EQ(std::this_thread::get_id(), caller);
        EXPECT_EQ(runtime->info().active_sessions, 1u);
        ++*calls;
        throw std::runtime_error("rejection callback failed");
      }));
  EXPECT_EQ(calls->load(), 2);
  EXPECT_FALSE(result(runtime->reset_session_async("session")));
  EXPECT_EQ(executor.open_attempts.load(), 2);
  EXPECT_FALSE(result(runtime->close_session_async("session")));
  runtime->shutdown();
  EXPECT_EQ(calls->load(), 2);
  EXPECT_EQ(executor.closed(), executor.opened());
}
#endif

TEST_F(ServingRuntimeTest, EmptyCallbacksDiscardNotificationNotOperation) {
  watch_callbacks();
  executor.opening.hold();
  start({1, 128, 4});
  runtime->open_session_async("session", {});
  ASSERT_TRUE(executor.opening.wait_for(1));
  runtime->reset_session_async("session", {});
  runtime->close_session_async("session", {});
  auto reopened = runtime->open_session_async("session");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(reopened)));
  EXPECT_EQ(executor.open_attempts.load(), 3);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  runtime->open_session_async("", {});
  runtime->shutdown();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(ServingRuntimeTest, SmallCallbackDestructionCanReenterInfo) {
  watch_callbacks();
  executor.opening.hold();
  start();
  // Keep this nothrow-copyable and small enough for libc++'s inline storage.
  struct SmallCallback {
    ServingRuntime* runtime;
    SmallCallbackState* state;

    ~SmallCallback() {
      (void)runtime->info();
      ++state->destructions;
    }

    void operator()(LifecycleResult error) const {
      EXPECT_FALSE(error);
      state->thread = std::this_thread::get_id();
      ++state->calls;
    }
  };
  static_assert(sizeof(SmallCallback) == 2 * sizeof(void*));
  static_assert(std::is_nothrow_copy_constructible<SmallCallback>::value);
  runtime->open_session_async(
      "session", SmallCallback{runtime.get(), &small_callback_state});
  ASSERT_TRUE(executor.opening.wait_for(1));
  const auto before_completion = small_callback_state.destructions.load();
  auto closed = runtime->close_session_async("session");
  executor.opening.release();
  EXPECT_FALSE(result(std::move(closed)));
  runtime->shutdown();
  EXPECT_GT(small_callback_state.destructions.load(), before_completion);
  EXPECT_EQ(small_callback_state.calls.load(), 1);
  EXPECT_NE(small_callback_state.thread, std::this_thread::get_id());
  for (auto engine : executor.threads()) {
    EXPECT_NE(small_callback_state.thread, engine);
  }
}

TEST_F(
    ServingRuntimeTest,
    ShutdownWaitsForCallbackAndCaptureAfterPermitRelease) {
  watch_callbacks();
  start({1, 128, 1});
  callback_gate.hold();
  auto cleaned = std::make_shared<std::atomic<bool>>(false);
  auto capture = std::shared_ptr<int>(new int(0), [this, cleaned](int* value) {
    cleanup_gate.arrive();
    *cleaned = true;
    delete value;
  });
  auto calls = std::make_shared<std::atomic<int>>(0);
  runtime->open_session_async("session", [this, capture, calls](auto error) {
    (void)capture;
    EXPECT_FALSE(error);
    EXPECT_EQ(runtime->info().active_sessions, 1u);
    ++*calls;
    callback_gate.arrive();
  });
  ASSERT_TRUE(callback_gate.wait_for(1));
  cleanup_gate.hold();
  capture.reset();
  // Admission succeeds while the previous callback still owns its captures.
  auto closed = runtime->close_session_async("session");
  EXPECT_EQ(
      closed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  auto stopped = std::make_shared<std::atomic<int>>(0);
  for (int i = 0; i < 2; ++i) {
    callback_callers.emplace_back([this, stopped] {
      runtime->shutdown();
      ++*stopped;
    });
  }
  ASSERT_TRUE(wait_until([this] { return !runtime->info().ready; }));
  EXPECT_EQ(stopped->load(), 0);
  EXPECT_FALSE(cleaned->load());
  callback_gate.release();
  ASSERT_TRUE(cleanup_gate.wait_for(1));
  EXPECT_EQ(stopped->load(), 0);
  EXPECT_FALSE(cleaned->load());
  cleanup_gate.release();
  ASSERT_TRUE(wait_until([stopped] { return stopped->load() == 2; }));
  expect_error(std::move(closed), ErrorCode::NotReady);
  EXPECT_TRUE(cleaned->load());
  EXPECT_EQ(calls->load(), 1);
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
  for (const auto& config :
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

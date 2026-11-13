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
#include <gtest/gtest.h>

#include <condition_variable>
#include <mutex>
#include <thread>

namespace {
using namespace executorch::extension::llm::batching;

class CheckpointExecutor
    : public executorch::extension::llm::batching::testing::FakeExecutor {
 public:
  bool fail_export = false;
  bool fail_restore = false;
  std::function<void()> after_restore;
  CheckpointResult<Checkpoint> export_checkpoint_to_cpu(
      SessionId,
      Position upto) override {
    if (fail_export) {
#if ET_HAS_EXCEPTIONS
      throw std::runtime_error("device export failed");
#else
      return CheckpointError{"device export failed"};
#endif
    }
    Checkpoint state;
    state.format = "test";
    state.position = upto;
    return state;
  }
  CheckpointResult<SessionId> restore_checkpoint_from_cpu(
      const Checkpoint& state,
      Position upto) override {
    if (fail_restore) {
#if ET_HAS_EXCEPTIONS
      throw std::runtime_error("device restore failed");
#else
      return CheckpointError{"device restore failed"};
#endif
    }
    if (state.format != "test" || upto > state.position)
      return CheckpointError{"incompatible"};
    auto id = open_session();
    if (!id)
      return CheckpointError{"capacity"};
    if (after_restore) {
      after_restore();
    }
    return *id;
  }
};

TEST(RunnerCheckpoint, RestoreUsesExplicitPositionWithoutSourcePendingToken) {
  CheckpointExecutor executor;
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto source = runner.open_session_async().get();
  ASSERT_TRUE(source);
  GenConfig config;
  config.max_new_tokens = 3;
  source->generate_async(std::vector<Token>{11, 12, 13}, config, {}).wait();
  EXPECT_EQ(source->position(), 5);
  auto result = source->export_checkpoint_to_cpu_async(3).get();
  ASSERT_TRUE(std::holds_alternative<Checkpoint>(result));
  auto checkpoint = std::make_shared<const Checkpoint>(
      std::get<Checkpoint>(std::move(result)));
  auto restored = runner.restore_checkpoint_from_cpu_async(checkpoint, 2).get();
  ASSERT_TRUE(std::holds_alternative<Session>(restored));
  auto session = std::get<Session>(std::move(restored));
  EXPECT_EQ(session.position(), 2);
  config.max_new_tokens = 1;
  session.generate_async(std::vector<Token>{99}, config, {}).wait();
  EXPECT_EQ(session.position(), 3);
  EXPECT_EQ(source->position(), 5);
  const auto seen = executor.seen();
  EXPECT_EQ(seen.back().effective_position(), 2);
  EXPECT_EQ(seen.back().size, 1);
  runner.shutdown();
  EXPECT_EQ(executor.open_count(), 0);
}

TEST(RunnerCheckpoint, InvalidPositionsAndBackendFailureLeaveSourceUsable) {
  CheckpointExecutor executor;
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto source = runner.open_session_async().get();
  ASSERT_TRUE(source);
  GenConfig config;
  config.max_new_tokens = 1;
  source->generate_async(std::vector<Token>{11, 12, 13}, config, {}).wait();
  for (Position p : {-1, 4}) {
    EXPECT_TRUE(std::holds_alternative<CheckpointError>(
        source->export_checkpoint_to_cpu_async(p).get()));
  }
  executor.fail_export = true;
  auto failed = source->export_checkpoint_to_cpu_async(3).get();
  ASSERT_TRUE(std::holds_alternative<CheckpointError>(failed));
  EXPECT_EQ(std::get<CheckpointError>(failed).message, "device export failed");
  executor.fail_export = false;
  EXPECT_TRUE(std::holds_alternative<Checkpoint>(
      source->export_checkpoint_to_cpu_async(3).get()));
  EXPECT_EQ(executor.open_count(), 1);
  runner.shutdown();
  EXPECT_TRUE(std::holds_alternative<CheckpointError>(
      source->export_checkpoint_to_cpu_async(3).get()));
}

TEST(RunnerCheckpoint, RestoreFailureDoesNotLeakSessions) {
  CheckpointExecutor executor;
  executor.capacity = 1;
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto source = runner.open_session_async().get();
  auto checkpoint = std::make_shared<Checkpoint>();
  checkpoint->format = "test";
  checkpoint->position = 5;
  EXPECT_TRUE(std::holds_alternative<CheckpointError>(
      runner.restore_checkpoint_from_cpu_async(checkpoint, 6).get()));
  EXPECT_TRUE(std::holds_alternative<CheckpointError>(
      runner.restore_checkpoint_from_cpu_async(checkpoint, 5).get()));
  EXPECT_EQ(executor.open_count(), 1);
  source.reset();
  auto restored = runner.restore_checkpoint_from_cpu_async(checkpoint, 5).get();
  ASSERT_TRUE(std::holds_alternative<Session>(restored));
  EXPECT_EQ(std::get<Session>(restored).position(), 5);
  runner.shutdown();
  EXPECT_EQ(executor.open_count(), 0);
}

TEST(RunnerCheckpoint, RestoreBackendFailurePreservesDiagnosticAndCanRetry) {
  CheckpointExecutor executor;
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto checkpoint = std::make_shared<Checkpoint>();
  checkpoint->format = "test";
  checkpoint->position = 5;
  executor.fail_restore = true;
  auto failed = runner.restore_checkpoint_from_cpu_async(checkpoint, 5).get();
  ASSERT_TRUE(std::holds_alternative<CheckpointError>(failed));
  EXPECT_EQ(std::get<CheckpointError>(failed).message, "device restore failed");
  EXPECT_EQ(executor.open_count(), 0);
  executor.fail_restore = false;
  EXPECT_TRUE(std::holds_alternative<Session>(
      runner.restore_checkpoint_from_cpu_async(checkpoint, 5).get()));
  runner.shutdown();
  EXPECT_EQ(executor.open_count(), 0);
}

TEST(RunnerCheckpoint, ShutdownDuringRestoreClosesAllocatedSession) {
  CheckpointExecutor executor;
  std::mutex mutex;
  std::condition_variable cv;
  bool entered = false, release = false;
  executor.after_restore = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cv.notify_all();
    cv.wait(lock, [&] { return release; });
  };
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto checkpoint = std::make_shared<Checkpoint>();
  checkpoint->format = "test";
  checkpoint->position = 5;
  auto restored = runner.restore_checkpoint_from_cpu_async(checkpoint, 5);
  {
    std::unique_lock<std::mutex> lock(mutex);
    EXPECT_TRUE(
        cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }));
  }
  auto stopping = std::async(std::launch::async, [&] { runner.shutdown(); });
  // Shutdown must stop admission before the in-flight executor call returns.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool stopped_admission = false;
  while (std::chrono::steady_clock::now() < deadline) {
    auto probe = runner.accepts_async(nullptr);
    if (probe.wait_for(std::chrono::milliseconds(1)) ==
        std::future_status::ready) {
      stopped_admission =
          probe.get() == AcceptanceResult{AcceptanceError::Unavailable};
      break;
    }
  }
  EXPECT_TRUE(stopped_admission);
  {
    std::lock_guard<std::mutex> lock(mutex);
    release = true;
  }
  cv.notify_all();
  stopping.get();
  auto result = restored.get();
  ASSERT_TRUE(std::holds_alternative<CheckpointError>(result));
  EXPECT_EQ(std::get<CheckpointError>(result).message, "runner is stopping");
  EXPECT_EQ(executor.open_count(), 0);
  EXPECT_EQ(executor.opened().size(), executor.closed().size());
}

TEST(RunnerCheckpoint, AbandonedRestoreFutureReleasesSession) {
  CheckpointExecutor executor;
  Runner runner(executor, DecodeFirstScheduler::create(8, 2, 4));
  auto checkpoint = std::make_shared<Checkpoint>();
  checkpoint->format = "test";
  checkpoint->position = 5;
  {
    auto abandoned = runner.restore_checkpoint_from_cpu_async(checkpoint, 5);
    abandoned.wait();
    // Let the engine release its promise before destroying the sole future.
    runner.accepts_async(nullptr).get();
    EXPECT_EQ(executor.open_count(), 1);
  }
  // The abandoned result owns the Session and queues its close on destruction.
  auto next = runner.open_session_async().get();
  ASSERT_TRUE(next);
  EXPECT_EQ(executor.open_count(), 1);
  EXPECT_EQ(executor.closed().size(), 1);
  runner.shutdown();
  EXPECT_EQ(executor.open_count(), 0);
}
} // namespace

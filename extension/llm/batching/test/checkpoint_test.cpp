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

namespace {
using namespace executorch::extension::llm::batching;

class CheckpointExecutor
    : public executorch::extension::llm::batching::testing::FakeExecutor {
 public:
  bool fail_export = false;
  CheckpointResult<Checkpoint> export_checkpoint_to_cpu(
      SessionId,
      Position upto) override {
    if (fail_export)
      throw std::runtime_error("device export failed");
    Checkpoint state;
    state.format = "test";
    state.position = upto;
    return state;
  }
  CheckpointResult<SessionId> restore_checkpoint_from_cpu(
      const Checkpoint& state,
      Position upto) override {
    if (state.format != "test" || upto > state.position)
      return CheckpointError{"incompatible"};
    auto id = open_session();
    if (!id)
      return CheckpointError{"capacity"};
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
} // namespace

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

#include <gtest/gtest.h>
#include <future>
#include <limits>
#include <string>
#include <thread>

namespace executorch::extension::llm::batching {
namespace {
constexpr std::chrono::seconds kTimeout{5};

struct Payload final : PreparedInput {
  inline static char kKind = 0;

  explicit Payload(std::vector<Token> rows = {11, 7, 7, 7, 22})
      : values(std::move(rows)), count(values.size()) {}
  std::vector<Token> values;
  std::size_t count;
  const void* kind() const override {
    return &kKind;
  }
  std::size_t size() const override {
    return count;
  }
};

struct WrongPayload final : PreparedInput {
  inline static char kKind = 0;

  const void* kind() const override {
    return &kKind;
  }
  std::size_t size() const override {
    return 1;
  }
};

// Reuse the text fake's sessions, sampling, gates and predictions. Only this
// adapter understands its opaque backing; recordings do not keep it alive.
struct RecordingExecutor : testing::FakeExecutor {
  std::vector<Input> slices;
  std::vector<std::vector<Token>> values;
  std::vector<std::weak_ptr<const void>> consumed;

  bool accepts(const PreparedInput& input) const override {
    return input.kind() == &Payload::kKind;
  }

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    BatchInput raw = batch;
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      const auto& input = batch.inputs[i];
      auto& translated = raw.inputs[i];
      if (const auto* prepared =
              std::get_if<PreparedInputPtr>(&input.payload)) {
        if (!accepts(**prepared)) {
          ADD_FAILURE() << "unsupported input reached execute";
          return false;
        }
        const auto& rows = static_cast<const Payload&>(**prepared).values;
        if (input.offset > rows.size() ||
            input.size > rows.size() - input.offset) {
          return false;
        }
        translated.payload = TokenInputPtr(*prepared, &rows);
      }
      slices.push_back(input);
      std::visit([](auto& owner) { owner.reset(); }, slices.back().payload);
      std::visit(
          [&](const auto& owner) { consumed.emplace_back(owner); },
          input.payload);
      const auto& tokens = *std::get<TokenInputPtr>(translated.payload);
      values.emplace_back(
          tokens.begin() + input.offset,
          tokens.begin() + input.offset + input.size);
    }
    return FakeExecutor::execute(raw, out);
  }
};

struct Harness {
  RecordingExecutor executor;
  Runner runner{executor, DecodeFirstScheduler::create(3, 1, 2)};
  ~Harness() {
    executor.release();
    runner.shutdown();
  }
  Session session() {
    auto future = runner.open_session_async();
    if (future.wait_for(kTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "open_session_async did not settle";
      return Session{};
    }
    auto result = future.get();
    EXPECT_TRUE(result);
    return result ? std::move(*result) : Session{};
  }
};

GenConfig config(int tokens = 1) {
  GenConfig result;
  result.max_new_tokens = tokens;
  return result;
}

TEST(PreparationTest, WarmOpaqueChunksAndOrdinaryRawText) {
  for (bool chunked : {false, true}) {
    SCOPED_TRACE(chunked);
    Harness h;
    auto session = h.session();
    std::vector<Token> cold_output;
    std::vector<Token> warm_output;
    auto cold = session.generate_async(
        GenerationInput{std::vector<Token>{1}},
        config(),
        [&](const GenerationUpdate& update) { cold_output = update.tokens; });
    cold.wait();
    auto warm = session.generate_async(
        std::vector<Token>{2}, config(), [&](const GenerationUpdate& update) {
          warm_output = update.tokens;
        });
    warm.wait();
    PreparedInputPtr opaque = std::make_shared<Payload>(
        chunked ? std::vector<Token>{11, 7, 7, 7, 22}
                : std::vector<Token>{11, 22});
    auto generation = session.generate_async(std::move(opaque), config(), {});
    generation.wait();
    h.runner.shutdown();
    EXPECT_EQ(cold.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(warm.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
    ASSERT_EQ(cold_output.size(), 1u);
    ASSERT_EQ(warm_output.size(), 1u);
    std::vector<std::vector<Token>> expected{
        {1}, {cold_output[0], 2}, {warm_output[0]}};
    if (chunked) {
      expected.insert(expected.end(), {{11, 7}, {7, 7}, {22}});
    } else {
      expected.push_back({11, 22});
    }
    EXPECT_EQ(h.executor.values, expected);
    // The pending raw token shares the first opaque batch. The longer backing
    // needs another batch for its remaining chunks.
    EXPECT_EQ(
        h.executor.batch_sizes(),
        chunked ? (std::vector<int>{1, 1, 2, 2}) : (std::vector<int>{1, 1, 2}));
    ASSERT_EQ(h.executor.slices.size(), expected.size());
    Position next = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto& slice = h.executor.slices[i];
      EXPECT_EQ(slice.sid, h.executor.slices.front().sid);
      EXPECT_EQ(
          std::holds_alternative<PreparedInputPtr>(slice.payload), i >= 3);
      EXPECT_EQ(slice.offset, i >= 3 ? 2 * (i - 3) : 0u);
      EXPECT_EQ(slice.position, i >= 3 ? 4 : (i == 2 ? 3 : i));
      EXPECT_EQ(slice.size, expected[i].size());
      EXPECT_EQ(slice.position + static_cast<Position>(slice.offset), next);
      EXPECT_EQ(slice.produce_output, i < 2 || i + 1 == expected.size());
      next += static_cast<Position>(slice.size);
    }
    EXPECT_EQ(session.position(), chunked ? 9 : 6);
    EXPECT_EQ(generation.metrics().n_prompt_tokens, chunked ? 5 : 2);
    EXPECT_EQ(generation.metrics().n_prefilled_tokens, chunked ? 6 : 3);
  }
}

TEST(PreparationTest, SharedOpaqueChunksKeepOwnershipAndUseRawFeedback) {
  Harness h;
  auto session = h.session();
  PreparedInputPtr opaque = std::make_shared<Payload>();
  std::weak_ptr<const PreparedInput> owner = opaque;
  std::vector<Token> output;
  auto handle = session.generate_async(
      std::move(opaque), config(2), [&](const GenerationUpdate& update) {
        output.insert(output.end(), update.tokens.begin(), update.tokens.end());
      });
  handle.wait();
  h.runner.shutdown();
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_EQ(output.size(), 2u);
  EXPECT_EQ(
      h.executor.values,
      (std::vector<std::vector<Token>>{{11, 7}, {7, 7}, {22}, {output[0]}}));
  EXPECT_EQ(h.executor.batch_sizes(), (std::vector<int>{1, 2, 1}));
  ASSERT_EQ(h.executor.slices.size(), 4u);
  ASSERT_EQ(h.executor.consumed.size(), 4u);
  for (std::size_t i = 0; i < 3; ++i) {
    const auto& slice = h.executor.slices[i];
    EXPECT_TRUE(std::holds_alternative<PreparedInputPtr>(slice.payload));
    EXPECT_EQ(slice.position, 0);
    EXPECT_EQ(slice.offset, 2 * i);
    EXPECT_EQ(slice.produce_output, i == 2);
    // Compare control blocks even when only weak owners remain.
    EXPECT_FALSE(h.executor.consumed[i].owner_before(owner));
    EXPECT_FALSE(owner.owner_before(h.executor.consumed[i]));
  }
  const auto& feedback = h.executor.slices.back();
  EXPECT_TRUE(std::holds_alternative<TokenInputPtr>(feedback.payload));
  EXPECT_EQ(feedback.position, 5);
  EXPECT_EQ(feedback.offset, 0u);
  EXPECT_TRUE(feedback.produce_output);
  EXPECT_EQ(session.position(), 6);
  EXPECT_EQ(handle.metrics().n_prefilled_tokens, 5);
  EXPECT_EQ(handle.metrics().n_prefill_steps, 2);
  EXPECT_EQ(handle.metrics().n_decode_steps, 1);
}

TEST(PreparationTest, DefaultExecutorRejectsOpaqueWithoutPoisoningSession) {
  testing::FakeExecutor executor;
  Runner runner{executor, DecodeFirstScheduler::create(3, 1, 2)};
  auto future = runner.open_session_async();
  ASSERT_EQ(future.wait_for(kTimeout), std::future_status::ready);
  auto session = future.get();
  ASSERT_TRUE(session);

  auto rejected = session->generate_async(
      PreparedInputPtr{std::make_shared<Payload>()}, config(), {});
  rejected.wait();
  EXPECT_EQ(rejected.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(session->position(), 0);
  EXPECT_TRUE(executor.seen().empty());
  EXPECT_FALSE(executor.has_sampling_state(executor.opened().front()));

  auto raw = session->generate_async(std::vector<Token>{1}, config(), {});
  raw.wait();
  runner.shutdown();
  EXPECT_EQ(raw.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(session->position(), 1);
}

TEST(PreparationTest, WrongKindDoesNotFailConcurrentRawGeneration) {
  Harness h;
  auto raw_session = h.session();
  auto bad_session = h.session();
  ASSERT_TRUE(raw_session.valid());
  ASSERT_TRUE(bad_session.valid());
  const auto raw_sid = h.executor.opened().front();

  h.executor.hold();
  auto raw =
      raw_session.generate_async(std::vector<Token>{1, 2}, config(2), {});
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (!h.executor.in_execute() &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(h.executor.in_execute());
  EXPECT_FALSE(raw.done());
  auto rejected = bad_session.generate_async(
      PreparedInputPtr{std::make_shared<WrongPayload>()}, config(), {});
  h.executor.release();
  rejected.wait();
  raw.wait();
  h.runner.shutdown();

  EXPECT_EQ(rejected.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(bad_session.position(), 0);
  EXPECT_EQ(raw.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(raw.metrics().n_generated_tokens, 2);
  EXPECT_EQ(raw_session.position(), 3);
  EXPECT_EQ(h.executor.batch_sizes(), (std::vector<int>{1, 1}));
  ASSERT_EQ(h.executor.slices.size(), 2u);
  for (const auto& slice : h.executor.slices) {
    EXPECT_EQ(slice.sid, raw_sid);
    EXPECT_TRUE(std::holds_alternative<TokenInputPtr>(slice.payload));
  }
}

TEST(PreparationTest, InvalidOpaqueMetadata) {
  auto empty = std::make_shared<Payload>(std::vector<Token>{});
  auto oversized = std::make_shared<Payload>();
  oversized->count = std::size_t{1} + std::numeric_limits<Position>::max();
  for (PreparedInputPtr input :
       {PreparedInputPtr{},
        PreparedInputPtr{empty},
        PreparedInputPtr{oversized}}) {
    SCOPED_TRACE(input ? std::to_string(input->size()) : "null");
    // This executor accepts Payload, so rejection must be due to its metadata.
    RecordingExecutor executor;
    Runner runner{executor, DecodeFirstScheduler::create(3, 1, 2)};
    auto future = runner.open_session_async();
    ASSERT_EQ(future.wait_for(kTimeout), std::future_status::ready);
    auto session = future.get();
    ASSERT_TRUE(session);
    auto handle = session->generate_async(std::move(input), config(), {});
    handle.wait();
    runner.shutdown();
    EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
    EXPECT_EQ(session->position(), 0);
    EXPECT_TRUE(executor.batch_sizes().empty());
    EXPECT_TRUE(executor.seen().empty());
  }
}
} // namespace
} // namespace executorch::extension::llm::batching

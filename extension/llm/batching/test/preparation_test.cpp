/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-Deprecated
#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/detail/input_validation.h>
#include <executorch/extension/llm/batching/runner.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>

#include <gtest/gtest.h>
#include <future>
#include <limits>
#include <stdexcept>

namespace executorch::extension::llm::batching {
namespace {
constexpr std::chrono::seconds kTimeout{5};

struct Gate {
  std::promise<void> entered;
  std::promise<void> released;
  std::future<void> arrival = entered.get_future();
  std::shared_future<void> departure = released.get_future().share();
  std::atomic<bool> open{false};
  void wait() {
    entered.set_value();
    EXPECT_EQ(departure.wait_for(kTimeout), std::future_status::ready);
  }
  void release() {
    if (!open.exchange(true)) {
      released.set_value();
    }
  }
  ~Gate() {
    release();
  }
};

struct Payload final : PreparedInput {
  std::vector<Token> values;
  std::size_t count = 0;
  std::size_t size() const override {
    return count;
  }
};

// Reuse the text fake's sessions, sampling, gates and deterministic
// predictions. Only this adapter understands its own prepared rows.
struct PreparingExecutor : testing::FakeExecutor {
  enum Reply { Valid, Null, Empty, Overflow, Failure, Throw };
  Reply reply = Valid;
  std::shared_ptr<Gate> gate;
  std::function<void()> after_prepare;
  std::vector<std::thread::id> threads;
  std::vector<Input> slices;
  std::vector<std::vector<Token>> values;
  std::vector<std::weak_ptr<const void>> consumed;
  std::vector<InputPayload> retained;
  std::weak_ptr<const PreparedInput> created;
  int preparations = 0;
  bool retain = false;
  bool fail_sampling = false;

  void note() {
    threads.push_back(std::this_thread::get_id());
  }
  bool initialize() override {
    note();
    return FakeExecutor::initialize();
  }
  bool prepare(const PreparationInput& input, PreparedInputPtr& out) override {
    note();
    ++preparations;
    if (gate) {
      gate->wait();
    }
    auto payload = std::make_shared<Payload>();
    for (const auto& segment : input.segments) {
      if (segment.is_tokens()) {
        const auto& tokens = segment.get_tokens();
        payload->values.insert(
            payload->values.end(), tokens.begin(), tokens.end());
      } else if (segment.is_image()) {
        const auto& image = segment.get_image();
        if (!image.is_uint8() || image.get_uint8_data().size() != 1) {
          return false;
        }
        payload->values.insert(
            payload->values.end(), 3, image.get_uint8_data()[0]);
      } else {
        return false;
      }
    }
    payload->count = reply == Empty ? 0 : payload->values.size();
    if (reply == Overflow) {
      payload->count = std::size_t{1} + std::numeric_limits<Position>::max();
    }
    out = reply == Null ? nullptr : payload;
    created = out;
#if ET_HAS_EXCEPTIONS
    if (reply == Throw) {
      throw std::runtime_error("prepare failed");
    }
#endif
    if (after_prepare) {
      after_prepare();
    }
    return reply != Failure;
  }
  void set_sampling(
      SessionId sid,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) override {
    note();
#if ET_HAS_EXCEPTIONS
    if (fail_sampling) {
      throw std::bad_alloc();
    }
#endif
    FakeExecutor::set_sampling(sid, params, seed);
  }
  bool execute(const BatchInput& batch, BatchOutput& out) override {
    note();
    if (!detail::validate_batch(batch)) {
      return false;
    }
    BatchInput raw = batch;
    for (auto& input : raw.inputs) {
      if (const auto* prepared =
              std::get_if<PreparedInputPtr>(&input.payload)) {
        const auto& rows = static_cast<const Payload&>(**prepared).values;
        if (input.offset > rows.size() ||
            input.size > rows.size() - input.offset) {
          return false;
        }
        input.payload = TokenInputPtr(*prepared, &rows);
      }
    }
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      const auto& input = batch.inputs[i];
      slices.push_back(input);
      slices.back().payload = TokenInputPtr{};
      std::visit(
          [&](const auto& owner) { consumed.emplace_back(owner); },
          input.payload);
      if (retain) {
        retained.push_back(input.payload);
      }
      const auto& tokens = *std::get<TokenInputPtr>(raw.inputs[i].payload);
      values.emplace_back(
          tokens.begin() + input.offset,
          tokens.begin() + input.offset + input.size);
    }
    return FakeExecutor::execute(raw, out);
  }
};

struct Harness {
  PreparingExecutor executor;
  Runner runner{executor, DecodeFirstScheduler::create(3, 1, 2)};
  ~Harness() {
    if (executor.gate) {
      executor.gate->release();
    }
    executor.release();
    runner.shutdown();
  }
  Session session() {
    auto result = runner.open_session_async().get();
    EXPECT_TRUE(result);
    return result ? std::move(*result) : Session{};
  }
};

PreparationInput input(bool image = true) {
  PreparationInput result;
  result.segments.emplace_back(std::vector<Token>{11});
  if (image) {
    result.segments.emplace_back(Image(std::vector<uint8_t>{7}, 1, 1, 1));
  }
  result.segments.emplace_back(std::vector<Token>{22});
  return result;
}

struct Completion {
  std::atomic<int> calls{0};
  std::promise<PreparedInputPtr> promise;
  std::future<PreparedInputPtr> future = promise.get_future();
  void complete(bool ok, PreparedInputPtr output) {
    EXPECT_EQ(ok, static_cast<bool>(output));
    if (calls.fetch_add(1) == 0) {
      promise.set_value(std::move(output));
    }
  }
  PreparedInputPtr take() {
    if (future.wait_for(kTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "preparation did not settle";
      return {};
    }
    return future.get();
  }
};

std::shared_ptr<Completion> prepare(
    Runner& runner,
    bool image = true,
    CancellationToken cancellation = {}) {
  auto completion = std::make_shared<Completion>();
  runner.prepare_async(
      input(image),
      std::move(cancellation),
      [completion](bool ok, PreparedInputPtr output) {
        completion->complete(ok, std::move(output));
      });
  return completion;
}

GenConfig config(int tokens = 1) {
  GenConfig result;
  result.max_new_tokens = tokens;
  return result;
}

TEST(PreparationTest, ExplicitTextPreparationAndOrdinaryRawText) {
  Harness h;
  auto session = h.session();
  auto cold = session.generate_async(
      GenerationInput{std::vector<Token>{1}}, config(), {});
  cold.wait();
  auto warm = session.generate_async(std::vector<Token>{2}, config(), {});
  warm.wait();
  auto prepared = prepare(h.runner, false)->take();
  ASSERT_TRUE(prepared);
  EXPECT_EQ(prepared->size(), 2u);
  auto generation = session.generate_async(
      GenerationInput{std::move(prepared)}, config(), {});
  generation.wait();
  h.runner.shutdown();
  EXPECT_EQ(cold.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(warm.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(h.executor.preparations, 1);
  EXPECT_EQ(
      h.executor.values[1].size(), 2u); // main's merged pending + raw suffix
  EXPECT_EQ(h.executor.values[1].back(), 2u);
  EXPECT_EQ(session.position(), 6);
}

TEST(PreparationTest, VariantRejectionReleasesBackingBeforeCallbacks) {
  for (bool closed : {false, true}) {
    Harness h;
    auto session = closed ? h.session() : Session{};
    GenerationInput input = prepare(h.runner)->take();
    std::weak_ptr<const PreparedInput> owner =
        std::get<PreparedInputPtr>(input);
    ASSERT_FALSE(owner.expired());
    h.runner.shutdown();
    int updates = 0;
    int settled = 0;
    auto handle = session.generate_async(
        std::move(input),
        config(),
        [&](const GenerationUpdate& update) {
          EXPECT_TRUE(owner.expired());
          EXPECT_EQ(
              update.finish_reason,
              closed ? FinishReason::Cancelled : FinishReason::Failed);
          ++updates;
        },
        [&] {
          EXPECT_TRUE(owner.expired());
          ++settled;
        });
    EXPECT_TRUE(handle.done());
    EXPECT_EQ(updates, 1);
    EXPECT_EQ(settled, 1);
  }
}

TEST(PreparationTest, MixedChunksRawFeedbackAndIndependentOwners) {
  for (bool retain : {false, true}) {
    Harness h;
    h.executor.retain = retain;
    auto session = h.session();
    auto prepared = prepare(h.runner)->take();
    ASSERT_TRUE(prepared);
    EXPECT_EQ(prepared->size(), 5u);
    std::weak_ptr<const PreparedInput> owner = prepared;
    const auto identity = prepared.get();
    auto terminals = std::make_shared<std::atomic<int>>(0);
    auto handle = session.generate_async(
        std::move(prepared),
        config(2),
        [owner, retain, terminals](const GenerationUpdate& update) {
          EXPECT_EQ(owner.expired(), !retain);
          *terminals += update.finish_reason.has_value();
        });
    handle.wait();
    h.runner.shutdown();
    EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(terminals->load(), 1);
    EXPECT_EQ(h.executor.preparations, 1);
    ASSERT_EQ(h.executor.values.size(), 4u);
    EXPECT_EQ(h.executor.values[0], (std::vector<Token>{11, 7}));
    EXPECT_EQ(h.executor.values[1], (std::vector<Token>{7, 7}));
    EXPECT_EQ(h.executor.values[2], (std::vector<Token>{22}));
    EXPECT_EQ(h.executor.values[3].size(), 1u);
    for (std::size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(h.executor.slices[i].offset, 2 * i);
      EXPECT_EQ(h.executor.slices[i].produce_output, i == 2);
      if (retain) {
        EXPECT_EQ(
            std::get<PreparedInputPtr>(h.executor.retained[i]).get(), identity);
      }
    }
    EXPECT_EQ(h.executor.slices.back().offset, 0u);
    EXPECT_EQ(session.position(), 6);
    EXPECT_EQ(handle.metrics().n_prefilled_tokens, 5);
    for (auto thread : h.executor.threads) {
      EXPECT_EQ(thread, h.executor.threads.front());
      EXPECT_NE(thread, std::this_thread::get_id());
    }
    h.executor.retained.clear();
    for (const auto& consumed : h.executor.consumed) {
      EXPECT_TRUE(consumed.expired());
    }
  }
}

TEST(PreparationTest, FailureAndCancellationPreserveWarmStateAndSettleOnce) {
  for (auto reply :
       {PreparingExecutor::Null,
        PreparingExecutor::Empty,
        PreparingExecutor::Overflow,
        PreparingExecutor::Failure
#if ET_HAS_EXCEPTIONS
        ,
        PreparingExecutor::Throw
#endif
       }) {
    Harness h;
    auto session = h.session();
    session.generate_async(std::vector<Token>{1}, config(), {}).wait();
    h.executor.reply = reply;
    auto completion = prepare(h.runner);
    EXPECT_FALSE(completion->take());
    EXPECT_TRUE(h.executor.created.expired());
    EXPECT_EQ(session.position(), 1);
    h.executor.reply = PreparingExecutor::Valid;
    auto retry = session.generate_async(std::vector<Token>{2}, config(), {});
    retry.wait();
    h.runner.shutdown();
    EXPECT_EQ(completion->calls.load(), 1);
    EXPECT_EQ(retry.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(session.position(), 3);
    EXPECT_EQ(h.executor.values.back().size(), 2u);
  }
  for (bool in_flight : {false, true}) {
    Harness h;
    h.executor.gate = std::make_shared<Gate>();
    auto cancellation = std::make_shared<std::atomic<bool>>(!in_flight);
    auto completion = prepare(h.runner, true, cancellation);
    if (in_flight) {
      ASSERT_EQ(
          h.executor.gate->arrival.wait_for(kTimeout),
          std::future_status::ready);
      cancellation->store(true);
      h.executor.gate->release();
    }
    EXPECT_FALSE(completion->take());
    h.runner.shutdown();
    EXPECT_EQ(completion->calls.load(), 1);
    EXPECT_EQ(h.executor.preparations, in_flight ? 1 : 0);
    EXPECT_TRUE(h.executor.created.expired());
  }
}

TEST(PreparationTest, QueuedPreparationCancellationAndReentrantShutdown) {
  Harness h;
  auto session = h.session();
  auto cancelled = std::make_shared<std::atomic<bool>>(false);
  auto first = std::make_shared<Completion>();
  auto second = std::make_shared<Completion>();
  auto capture = std::make_shared<int>(1);
  std::weak_ptr<int> released = capture;
  // Both commands are posted before returning to the engine loop. The first
  // selected preparation cancels the second while it is in the scheduler.
  h.executor.after_prepare = [cancelled] { cancelled->store(true); };
  auto generation = session.generate_async(
      std::vector<Token>{1},
      config(),
      [&, first, second, capture = std::move(capture)](
          const GenerationUpdate&) {
        h.runner.prepare_async(
            input(), {}, [first](bool ok, PreparedInputPtr output) {
              first->complete(ok, std::move(output));
            });
        h.runner.prepare_async(
            input(),
            cancelled,
            [second, capture](bool ok, PreparedInputPtr output) {
              second->complete(ok, std::move(output));
            });
      });
  generation.wait();
  EXPECT_TRUE(first->take());
  EXPECT_FALSE(second->take());
  auto stopped = std::make_shared<Completion>();
  h.runner.prepare_async(
      input(false), {}, [&, stopped](bool ok, PreparedInputPtr output) {
        stopped->complete(ok, std::move(output));
        h.runner.shutdown();
      });
  EXPECT_TRUE(stopped->take());
  h.runner.shutdown();
  EXPECT_TRUE(released.expired());
  EXPECT_EQ(first->calls.load(), 1);
  EXPECT_EQ(second->calls.load(), 1);
  EXPECT_EQ(stopped->calls.load(), 1);
  EXPECT_EQ(h.executor.preparations, 2);
  auto rejected = prepare(h.runner);
  EXPECT_FALSE(rejected->take());
  EXPECT_EQ(rejected->calls.load(), 1);
}

TEST(PreparationTest, QueuedPreparedGenerationOwnsBackingAndCancels) {
  for (bool close : {false, true}) {
    Harness h;
    auto session = h.session();
    auto prepared = prepare(h.runner)->take();
    std::weak_ptr<const PreparedInput> owner = prepared;
    h.executor.gate = std::make_shared<Gate>();
    auto blocker = prepare(h.runner);
    ASSERT_EQ(
        h.executor.gate->arrival.wait_for(kTimeout), std::future_status::ready);
    auto terminal = std::make_shared<std::atomic<int>>(0);
    auto handle = session.generate_async(
        std::move(prepared),
        config(),
        [owner, terminal](const GenerationUpdate& update) {
          EXPECT_TRUE(owner.expired());
          EXPECT_EQ(update.finish_reason, FinishReason::Cancelled);
          ++*terminal;
        });
    EXPECT_FALSE(owner.expired());
    if (close) {
      session = Session{};
    } else {
      handle.cancel();
    }
    h.executor.gate->release();
    EXPECT_TRUE(blocker->take());
    handle.wait();
    h.runner.shutdown();
    EXPECT_EQ(terminal->load(), 1);
    EXPECT_EQ(handle.finish_reason(), FinishReason::Cancelled);
    EXPECT_TRUE(h.executor.values.empty());
  }
}

#if ET_HAS_EXCEPTIONS
TEST(
    PreparationTest,
    SamplingFailureReleasesStagedInputAndPreservesPendingToken) {
  Harness h;
  auto session = h.session();
  session.generate_async(std::vector<Token>{1}, config(), {}).wait();
  auto prepared = prepare(h.runner)->take();
  std::weak_ptr<const PreparedInput> owner = prepared;
  h.executor.fail_sampling = true;
  auto failed = session.generate_async(
      std::move(prepared), config(), [owner](const GenerationUpdate& update) {
        EXPECT_TRUE(owner.expired());
        EXPECT_EQ(update.finish_reason, FinishReason::Failed);
        throw std::runtime_error("callback failed");
      });
  failed.wait();
  EXPECT_EQ(failed.error_message(), "callback failed");
  EXPECT_EQ(session.position(), 1);
  h.executor.fail_sampling = false;
  auto retry = session.generate_async(std::vector<Token>{2}, config(), {});
  retry.wait();
  h.runner.shutdown();
  EXPECT_EQ(retry.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(session.position(), 3);
  EXPECT_EQ(h.executor.values.back().size(), 2u);
}
#endif

TEST(PreparationTest, ShutdownDiscardsInflightAndQueuedPreparation) {
  Harness h;
  auto session = h.session();
  auto first = std::make_shared<Completion>();
  auto second = std::make_shared<Completion>();
  // Inject the stop after preparation has produced output but before return.
  h.executor.after_prepare = [&] { h.runner.shutdown(); };
  auto generation = session.generate_async(
      std::vector<Token>{1},
      config(),
      [&, first, second](const GenerationUpdate&) {
        for (const auto& completion : {first, second}) {
          h.runner.prepare_async(
              input(), {}, [completion](bool ok, PreparedInputPtr output) {
                completion->complete(ok, std::move(output));
              });
        }
      });
  generation.wait();
  EXPECT_FALSE(first->take());
  EXPECT_FALSE(second->take());
  h.runner.shutdown();
  EXPECT_EQ(first->calls.load(), 1);
  EXPECT_EQ(second->calls.load(), 1);
  EXPECT_EQ(h.executor.preparations, 1);
  EXPECT_TRUE(h.executor.created.expired());
}

TEST(PreparationTest, OwnedBackingOutlivesRunnerAndDefaultHookRefusesIt) {
  PreparedInputPtr owner;
  {
    Harness h;
    owner = prepare(h.runner)->take();
    ASSERT_TRUE(owner);
  }
  EXPECT_EQ(owner->size(), 5u);
  testing::FakeExecutor executor;
  EXPECT_FALSE(executor.prepare(input(false), owner));
  EXPECT_FALSE(owner);
  executor.fail_initialize = true;
  Runner runner(executor, DecodeFirstScheduler::create());
  auto completion = prepare(runner);
  EXPECT_FALSE(completion->take());
  runner.shutdown();
  EXPECT_EQ(completion->calls.load(), 1);
}

TEST(PreparationTest, StructuralBounds) {
  auto rows = std::make_shared<Payload>();
  rows->count = 4;
  TokenInputPtr tokens = std::make_shared<const std::vector<Token>>(4, 1);
  for (auto payload :
       {InputPayload{PreparedInputPtr{rows}}, InputPayload{tokens}}) {
    BatchInput batch{{Input{1, false, 1, 3, payload, 0}}};
    EXPECT_TRUE(detail::validate_batch(batch));
    for (auto size :
         {std::size_t{0},
          std::size_t{4},
          std::numeric_limits<std::size_t>::max()}) {
      batch.inputs[0].size = size;
      EXPECT_FALSE(detail::validate_batch(batch));
    }
    batch.inputs[0].size = 1;
    batch.inputs[0].position = std::numeric_limits<Position>::max();
    EXPECT_FALSE(detail::validate_batch(batch));
    batch.inputs[0].position = -2;
    EXPECT_FALSE(detail::validate_batch(batch));
    batch.inputs[0].payload = PreparedInputPtr{};
    EXPECT_FALSE(detail::validate_batch(batch));
  }
}
} // namespace
} // namespace executorch::extension::llm::batching

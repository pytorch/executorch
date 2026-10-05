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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace executorch::extension::llm::batching {
namespace {

constexpr std::chrono::seconds kTimeout{5};

class Gate {
 public:
  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
    entered_ = false;
  }

  void enter() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!held_) {
      return;
    }
    entered_ = true;
    cv_.notify_all();
    EXPECT_TRUE(cv_.wait_for(lock, kTimeout, [this] { return !held_; }))
        << "test did not release an engine gate";
  }

  bool wait_for_entry() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, kTimeout, [this] { return entered_; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = false;
    cv_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool held_ = false;
  bool entered_ = false;
};

PreparationConfig test_preparation_config(std::size_t slots = 4) {
  PreparationConfig config;
  config.max_positions = 32;
  config.max_retained_bytes = 1024;
  config.max_workspace_bytes = 4096;
  // Each fixture slot covers one worst-case output and its small source.
  config.max_total_retained_bytes = slots * (config.max_retained_bytes + 512);
  config.max_images = 1;
  return config;
}

PreparationInput text_input(std::vector<Token> tokens = {11, 12}) {
  PreparationInput input;
  input.segments.emplace_back(std::move(tokens));
  return input;
}

PreparationInput image_input() {
  PreparationInput input;
  input.segments.emplace_back(std::vector<Token>{10, 11});
  input.segments.emplace_back(Image(std::vector<uint8_t>{7}, 1, 1, 1));
  input.segments.emplace_back(std::vector<Token>{20, 21, 22, 23, 24});
  return input;
}

GenConfig generation_config(std::int32_t count = 1) {
  GenConfig config;
  config.max_new_tokens = count;
  return config;
}

enum class PreparationReply {
  Valid,
  Null,
  Empty,
  ForeignTag,
  TooManyPositions,
  TooManyBytes,
  FailureWithOutput,
#if ET_HAS_EXCEPTIONS
  ExceptionWithOutput,
#endif
};

// Neither preparation nor feedback uses TokenPreparedInput. Only this executor
// understands the payload, including the three logical positions of an image.
class PrivateExecutor final : public Executor {
 public:
  struct Slice {
    Position base;
    std::size_t offset;
    std::size_t size;
    bool produce_output;
    bool feedback;
    const PreparedInput* identity;
    std::vector<std::int64_t> values;

    std::int64_t start() const {
      return static_cast<std::int64_t>(base) +
          static_cast<std::int64_t>(offset);
    }
  };

  explicit PrivateExecutor(PreparationConfig config = test_preparation_config())
      : Executor(config) {}

  bool initialize() override {
    note_call();
    initialize_gate.enter();
    return true;
  }

  bool prepare(const PreparationInput& input, PreparedInputPtr& out) override {
    note_call();
    ++prepare_calls;
    preparation_gate.enter();
    std::vector<std::int64_t> values;
    for (const auto& segment : input.segments) {
      if (segment.is_tokens()) {
        for (auto token : segment.get_tokens()) {
          values.push_back(static_cast<std::int64_t>(token));
        }
      } else if (segment.is_image()) {
        const auto& image = segment.get_image();
        if (!image.is_uint8() || image.get_uint8_data().size() != 1) {
          return false;
        }
        const auto pixel = image.get_uint8_data()[0];
        values.insert(values.end(), {-pixel, -pixel - 1, -pixel - 2});
      } else {
        return false;
      }
    }
    const auto reply = next_reply.exchange(PreparationReply::Valid);
    if (reply == PreparationReply::Null) {
      out.reset();
      return true;
    }
    auto count = values.size();
    auto bytes = std::size_t{512};
    if (reply == PreparationReply::Empty) {
      count = 0;
    } else if (reply == PreparationReply::TooManyPositions) {
      count = preparation_config().max_positions + 1;
    } else if (reply == PreparationReply::TooManyBytes) {
      bytes = preparation_config().max_retained_bytes + 1;
    }
    out = std::make_shared<const Payload>(
        std::move(values),
        count,
        bytes,
        false,
        reply == PreparationReply::ForeignTag);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      created_.push_back(out);
    }
#if ET_HAS_EXCEPTIONS
    if (reply == PreparationReply::ExceptionWithOutput) {
      throw std::runtime_error("preparation failed after creating output");
    }
#endif
    if (after_prepare) {
      after_prepare();
    }
    return reply != PreparationReply::FailureWithOutput;
  }

  bool wrap_tokens(
      std::shared_ptr<const std::vector<Token>> tokens,
      PreparedInputPtr& out) override {
    note_call();
    ++wrap_calls;
    if (!tokens || tokens->empty()) {
      return false;
    }
    const auto count = tokens->size();
    std::vector<std::int64_t> values(tokens->begin(), tokens->end());
    out = std::make_shared<const Payload>(
        std::move(values), count, 512, true, false, std::move(tokens));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      created_.push_back(out);
    }
    return !fail_wrap.load();
  }

  bool accepts(const PreparedInput& input) const override {
    note_call();
    return input.compatibility_tag() == Payload::tag();
  }

  std::optional<SessionId> open_session() override {
    note_call();
    const auto sid = next_session_++;
    histories_.emplace(sid, std::vector<std::int64_t>{});
    return sid;
  }

  void close_session(SessionId sid) override {
    note_call();
    histories_.erase(sid);
  }

  void set_sampling(
      SessionId,
      const SamplingParams&,
      std::optional<std::uint64_t>) override {
    note_call();
    ++sampling_calls;
#if ET_HAS_EXCEPTIONS
    if (fail_sampling_once.exchange(false)) {
      throw std::bad_alloc();
    }
#endif
    ++sampling_updates;
  }

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    note_call();
    if (!validate_batch(batch)) {
      return false;
    }
    execution_gate.enter();
    auto histories = histories_;
    std::vector<Slice> slices;
    for (const auto& input : batch.inputs) {
      const auto& payload = static_cast<const Payload&>(*input.prepared);
      const auto start = static_cast<std::int64_t>(input.position) +
          static_cast<std::int64_t>(input.offset);
      auto history = histories.find(input.sid);
      if (history == histories.end() || start < 0 ||
          static_cast<std::size_t>(start) > history->second.size() ||
          input.offset > payload.values.size() ||
          input.size > payload.values.size() - input.offset) {
        return false;
      }
      std::vector<std::int64_t> values(
          payload.values.begin() + input.offset,
          payload.values.begin() + input.offset + input.size);
      history->second.resize(static_cast<std::size_t>(start));
      history->second.insert(
          history->second.end(), values.begin(), values.end());
      slices.push_back(
          {input.position,
           input.offset,
           input.size,
           input.produce_output,
           payload.feedback,
           input.prepared.get(),
           std::move(values)});
    }
    histories_ = std::move(histories);
    if (retain_inputs) {
      for (const auto& input : batch.inputs) {
        retained_inputs.push_back(input.prepared);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      slices_.insert(slices_.end(), slices.begin(), slices.end());
    }
    out.outputs.clear();
    for (const auto& input : batch.inputs) {
      if (input.produce_output) {
        out.outputs.emplace_back(Output{input.sid, {next_prediction_++}});
      } else {
        out.outputs.emplace_back(std::nullopt);
      }
    }
    return true;
  }

  std::vector<Slice> slices() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slices_;
  }

  std::vector<std::weak_ptr<const PreparedInput>> created() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return created_;
  }

  std::vector<std::thread::id> threads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return threads_;
  }

  Gate initialize_gate;
  Gate preparation_gate;
  Gate execution_gate;
  std::atomic<int> prepare_calls{0};
  std::atomic<int> wrap_calls{0};
  std::atomic<bool> fail_wrap{false};
  std::atomic<int> sampling_calls{0};
  std::atomic<int> sampling_updates{0};
#if ET_HAS_EXCEPTIONS
  std::atomic<bool> fail_sampling_once{false};
#endif
  bool retain_inputs = false;
  std::vector<PreparedInputPtr> retained_inputs;
  std::function<void()> after_prepare;
  std::atomic<PreparationReply> next_reply{PreparationReply::Valid};

 private:
  class Payload final : public PreparedInput {
   public:
    Payload(
        std::vector<std::int64_t> values,
        std::size_t count,
        std::size_t bytes,
        bool feedback,
        bool foreign,
        std::shared_ptr<const std::vector<Token>> tokens = {})
        : values(std::move(values)),
          feedback(feedback),
          count_(count),
          bytes_(bytes),
          foreign_(foreign),
          tokens_(std::move(tokens)) {}

    std::size_t position_count() const override {
      return count_;
    }
    std::size_t retained_bytes() const override {
      return bytes_;
    }
    static const void* tag() {
      static const char identity = 0;
      return &identity;
    }
    const void* compatibility_tag() const override {
      static const char foreign_identity = 0;
      return foreign_ ? &foreign_identity : tag();
    }

    const std::vector<std::int64_t> values;
    const bool feedback;

   private:
    const std::size_t count_;
    const std::size_t bytes_;
    const bool foreign_;
    const std::shared_ptr<const std::vector<Token>> tokens_;
  };

  void note_call() const {
    std::lock_guard<std::mutex> lock(mutex_);
    threads_.push_back(std::this_thread::get_id());
  }

  mutable std::mutex mutex_;
  mutable std::vector<std::thread::id> threads_;
  std::vector<Slice> slices_;
  std::vector<std::weak_ptr<const PreparedInput>> created_;
  std::map<SessionId, std::vector<std::int64_t>> histories_;
  SessionId next_session_ = 1;
  Token next_prediction_ = 1000;
};

struct Harness {
  explicit Harness(PrivateExecutor& executor)
      : executor(executor),
        runner(executor, DecodeFirstScheduler::create(3, 1, 2)) {}

  ~Harness() {
    executor.initialize_gate.release();
    executor.preparation_gate.release();
    executor.execution_gate.release();
    runner.shutdown();
  }

  PrivateExecutor& executor;
  Runner runner;
};

struct PreparationResult {
  bool ok = false;
  PreparedInputPtr prepared;
  std::thread::id thread;
};

struct PreparationCompletion {
  std::atomic<int> calls{0};
  std::promise<PreparationResult> promise;
  std::future<PreparationResult> future = promise.get_future();

  void complete(bool ok, PreparedInputPtr prepared) {
    if (calls.fetch_add(1) == 0) {
      promise.set_value({ok, std::move(prepared), std::this_thread::get_id()});
    }
  }

  PreparationResult take() {
    if (future.wait_for(kTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "preparation did not complete";
      return {};
    }
    return future.get();
  }
};

std::shared_ptr<PreparationCompletion> prepare(
    Runner& runner,
    PreparationInput input = text_input(),
    CancellationToken cancellation = {}) {
  auto completion = std::make_shared<PreparationCompletion>();
  runner.prepare_async(
      std::move(input),
      std::move(cancellation),
      [completion](bool ok, PreparedInputPtr output) {
        completion->complete(ok, std::move(output));
      });
  return completion;
}

Session open_session(Runner& runner) {
  auto future = runner.open_session_async();
  if (future.wait_for(kTimeout) != std::future_status::ready) {
    ADD_FAILURE() << "open_session_async did not complete";
    return {};
  }
  auto session = future.get();
  EXPECT_TRUE(session.has_value());
  return session ? std::move(*session) : Session{};
}

struct GenerationCompletion {
  std::vector<Token> tokens;
  int terminal_calls = 0;
  std::atomic<int> settled_calls{0};
  std::promise<void> promise;
  std::future<void> future = promise.get_future();

  GenerationCallback callback() {
    return [this](const GenerationUpdate& update) {
      tokens.insert(tokens.end(), update.tokens.begin(), update.tokens.end());
      terminal_calls += update.finish_reason.has_value() ? 1 : 0;
    };
  }

  std::function<void()> settled() {
    return [this] {
      if (settled_calls.fetch_add(1) == 0) {
        promise.set_value();
      }
    };
  }

  bool wait() {
    return future.wait_for(kTimeout) == std::future_status::ready;
  }
};

#if ET_HAS_EXCEPTIONS
TEST(PreparationTest, SamplingFailureSettlesHandoffAndPreservesSessions) {
  for (bool raw : {false, true}) {
    for (bool warm : {false, true}) {
      for (bool throw_callbacks : {false, true}) {
        SCOPED_TRACE(
            ::testing::Message() << "raw=" << raw << " warm=" << warm
                                 << " throw_callbacks=" << throw_callbacks);
        auto config = test_preparation_config();
        std::vector<Token> probe_tokens{1};
        probe_tokens.reserve(512);
        auto probe = text_input(std::move(probe_tokens));
        // This source needs the entire budget. Admit it from the terminal
        // callback to detect any remaining input, carried, or feedback charge.
        config.max_workspace_bytes = 16 * 1024;
        config.max_total_retained_bytes =
            config.max_retained_bytes + *config.input_retained_bytes(probe);
        PrivateExecutor executor(config);
        GenerationCompletion initial;
        GenerationCompletion peer_initial;
        GenerationCompletion failed;
        GenerationCompletion recovered;
        GenerationCompletion peer_recovered;
        auto probe_completion = std::make_shared<PreparationCompletion>();
        Harness harness(executor);
        auto session = open_session(harness.runner);
        auto peer = open_session(harness.runner);
        ASSERT_TRUE(session.valid());
        ASSERT_TRUE(peer.valid());
        if (warm) {
          auto handle = session.generate_async(
              std::vector<Token>{7},
              generation_config(),
              initial.callback(),
              initial.settled());
          ASSERT_TRUE(initial.wait());
          ASSERT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
        }
        auto peer_handle = peer.generate_async(
            std::vector<Token>{71},
            generation_config(),
            peer_initial.callback(),
            peer_initial.settled());
        ASSERT_TRUE(peer_initial.wait());
        ASSERT_EQ(peer_handle.finish_reason(), FinishReason::NewTokenLimit);
        const auto prior_slices = executor.slices().size();
        const auto prior_sampling = executor.sampling_updates.load();
        const auto prior_wraps = executor.wrap_calls.load();

        PreparedInputPtr prepared;
        if (!raw) {
          auto result = prepare(harness.runner, image_input())->take();
          ASSERT_TRUE(result.ok);
          prepared = std::move(result.prepared);
        }
        executor.fail_sampling_once = true;
        auto callback = [&](const GenerationUpdate& update) {
          failed.callback()(update);
          EXPECT_EQ(update.finish_reason, FinishReason::Failed);
          EXPECT_TRUE(update.error_message.empty());
          for (const auto& owner : executor.created()) {
            EXPECT_TRUE(owner.expired())
                << "staged backing must be released before terminal delivery";
          }
          harness.runner.prepare_async(
              std::move(probe),
              {},
              [probe_completion](bool ok, PreparedInputPtr output) {
                probe_completion->complete(ok, std::move(output));
              });
          if (throw_callbacks) {
            throw std::runtime_error("terminal callback failed");
          }
        };
        auto settled = [&] {
          failed.settled()();
          if (throw_callbacks) {
            throw std::runtime_error("settlement callback failed");
          }
        };
        auto handle = raw ? session.generate_async(
                                std::vector<Token>{11, 12, 13, 14, 15},
                                generation_config(2),
                                callback,
                                settled)
                          : session.generate_async(
                                std::move(prepared),
                                3,
                                5,
                                generation_config(2),
                                callback,
                                settled);
        ASSERT_TRUE(failed.wait()) << "pre-install failure stranded the handle";
        EXPECT_TRUE(handle.done());
        EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
        EXPECT_EQ(
            handle.error_message(),
            throw_callbacks ? "terminal callback failed" : "");
        EXPECT_EQ(failed.terminal_calls, 1);
        EXPECT_EQ(failed.settled_calls.load(), 1);
        EXPECT_TRUE(failed.tokens.empty());
        EXPECT_EQ(session.position(), warm ? 1 : 0);
        EXPECT_EQ(peer.position(), 1);
        EXPECT_EQ(executor.slices().size(), prior_slices);
        EXPECT_EQ(executor.sampling_calls.load(), prior_sampling + 1);
        EXPECT_EQ(executor.sampling_updates.load(), prior_sampling);
        EXPECT_EQ(executor.wrap_calls.load(), prior_wraps + (warm ? 1 : 0));
        EXPECT_EQ(handle.metrics().n_prefilled_tokens, 0);
        EXPECT_EQ(handle.metrics().n_generated_tokens, 0);
        auto probe_result = probe_completion->take();
        ASSERT_TRUE(probe_result.ok) << "staging leaked a reservation";
        probe_result.prepared.reset();

        auto retry = session.generate_async(
            std::vector<Token>{21, 22},
            generation_config(2),
            recovered.callback(),
            recovered.settled());
        ASSERT_TRUE(recovered.wait());
        EXPECT_EQ(retry.finish_reason(), FinishReason::NewTokenLimit);
        EXPECT_EQ(session.position(), warm ? 5 : 3);
        auto slices = executor.slices();
        ASSERT_GT(slices.size(), prior_slices);
        if (warm) {
          EXPECT_EQ(
              slices[prior_slices].values,
              std::vector<std::int64_t>(
                  initial.tokens.begin(), initial.tokens.end()));
          EXPECT_TRUE(slices[prior_slices].feedback);
          EXPECT_FALSE(slices[prior_slices].produce_output);
          EXPECT_EQ(slices[prior_slices].start(), 1);
        }
        const auto before_peer = slices.size();
        auto peer_retry = peer.generate_async(
            std::vector<Token>{72},
            generation_config(2),
            peer_recovered.callback(),
            peer_recovered.settled());
        ASSERT_TRUE(peer_recovered.wait());
        EXPECT_EQ(peer_retry.finish_reason(), FinishReason::NewTokenLimit);
        EXPECT_EQ(peer.position(), 4);
        harness.runner.shutdown();
        slices = executor.slices();
        ASSERT_GT(slices.size(), before_peer);
        EXPECT_EQ(
            slices[before_peer].values,
            std::vector<std::int64_t>(
                peer_initial.tokens.begin(), peer_initial.tokens.end()));
        EXPECT_TRUE(slices[before_peer].feedback);
        EXPECT_FALSE(slices[before_peer].produce_output);
        EXPECT_EQ(slices[before_peer].start(), 1);
        EXPECT_EQ(failed.terminal_calls, 1);
        EXPECT_EQ(failed.settled_calls.load(), 1);
        EXPECT_EQ(recovered.terminal_calls, 1);
        EXPECT_EQ(recovered.settled_calls.load(), 1);
        EXPECT_EQ(peer_recovered.terminal_calls, 1);
        EXPECT_EQ(peer_recovered.settled_calls.load(), 1);
        const auto metrics = harness.runner.metrics();
        EXPECT_EQ(metrics.generations_started, warm ? 5u : 4u);
        EXPECT_EQ(metrics.generations_completed, metrics.generations_started);
        EXPECT_EQ(metrics.finished_failed, 1u);
        EXPECT_EQ(metrics.finished_token_limit, warm ? 4u : 3u);
      }
    }
  }
}
#endif

// Logical positions are compressed into a scalar. This executor never allocates
// storage proportional to a range, even when it spans nearly INT32_MAX slots.
class CompressedExecutor final : public Executor {
 public:
  class Payload final : public PreparedInput {
   public:
    explicit Payload(std::size_t count) : count_(count) {}
    std::size_t position_count() const override {
      return count_;
    }
    std::size_t retained_bytes() const override {
      return sizeof(Payload);
    }
    static const void* tag() {
      static const char identity = 0;
      return &identity;
    }
    const void* compatibility_tag() const override {
      return tag();
    }

   private:
    const std::size_t count_;
  };

  CompressedExecutor() : Executor(config()) {}

  bool accepts(const PreparedInput& input) const override {
    return input.compatibility_tag() == Payload::tag();
  }
  std::optional<SessionId> open_session() override {
    return 1;
  }
  void close_session(SessionId) override {}
  void set_sampling(
      SessionId,
      const SamplingParams&,
      std::optional<std::uint64_t>) override {}

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    if (!validate_batch(batch)) {
      return false;
    }
    out.outputs.clear();
    for (const auto& input : batch.inputs) {
      const auto start = static_cast<std::int64_t>(input.position) +
          static_cast<std::int64_t>(input.offset);
      if (start != static_cast<std::int64_t>(committed)) {
        return false;
      }
      committed += input.size;
      slices.push_back(input);
      slices.back().prepared.reset();
      if (input.produce_output) {
        ++predictions;
        out.outputs.emplace_back(Output{input.sid, {1000}});
      } else {
        out.outputs.emplace_back(std::nullopt);
      }
    }
    return true;
  }

  std::size_t committed = 0;
  int predictions = 0;
  std::vector<Input> slices;

 private:
  static PreparationConfig config() {
    auto config = test_preparation_config(1);
    config.max_positions = std::numeric_limits<Position>::max();
    return config;
  }
};

TEST(PreparationTest, LargeCompressedRangeUsesTwoChunksWithoutCursorOverflow) {
  constexpr auto count =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - 1;
  constexpr std::size_t chunk = std::size_t{1} << 30;
  for (std::size_t offset : {0u, 1u}) {
    SCOPED_TRACE(offset);
    CompressedExecutor executor;
    GenerationCompletion completion;
    Runner runner(executor, DecodeFirstScheduler::create(chunk + 1, 1, chunk));
    auto session = open_session(runner);
    ASSERT_TRUE(session.valid());
    PreparedInputPtr prepared =
        std::make_shared<const CompressedExecutor::Payload>(count + offset);
    std::weak_ptr<const PreparedInput> owner = prepared;
    auto handle = session.generate_async(
        std::move(prepared),
        offset,
        count,
        generation_config(),
        completion.callback(),
        completion.settled());
    ASSERT_TRUE(completion.wait());
    runner.shutdown();
    EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(completion.terminal_calls, 1);
    EXPECT_EQ(completion.settled_calls.load(), 1);
    EXPECT_EQ(completion.tokens, (std::vector<Token>{1000}));
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(session.position(), count);
    EXPECT_EQ(executor.committed, count);
    EXPECT_EQ(executor.predictions, 1);
    ASSERT_EQ(executor.slices.size(), 2u);
    const auto& first = executor.slices[0];
    const auto& last = executor.slices[1];
    EXPECT_EQ(first.position, -static_cast<Position>(offset));
    EXPECT_EQ(last.position, first.position);
    EXPECT_EQ(first.offset, offset);
    EXPECT_EQ(first.size, chunk);
    EXPECT_FALSE(first.produce_output);
    EXPECT_EQ(last.offset, offset + chunk);
    EXPECT_EQ(last.size, count - chunk);
    EXPECT_TRUE(last.produce_output);
    EXPECT_EQ(handle.metrics().n_prompt_tokens, count);
    EXPECT_EQ(handle.metrics().n_prefilled_tokens, count);
    EXPECT_EQ(handle.metrics().n_generated_tokens, 1);
    EXPECT_EQ(runner.metrics().generations_started, 1u);
    EXPECT_EQ(runner.metrics().generations_completed, 1u);
  }
}

TEST(PreparationTest, PrivateImageBackingExecutesOnlySelectedInteriorRange) {
  PrivateExecutor executor;
  GenerationCompletion completion;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  ASSERT_TRUE(session.valid());
  auto prepared = prepare(harness.runner, image_input())->take();
  ASSERT_TRUE(prepared.ok);
  ASSERT_NE(prepared.prepared, nullptr);
  ASSERT_EQ(prepared.prepared->position_count(), 10u);

  // Select the last two image positions and two text positions, excluding both
  // ends.
  auto handle = session.generate_async(
      prepared.prepared,
      3,
      4,
      generation_config(),
      completion.callback(),
      completion.settled());
  ASSERT_TRUE(completion.wait());
  harness.runner.shutdown();

  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(completion.tokens, (std::vector<Token>{1000}));
  EXPECT_EQ(completion.terminal_calls, 1);
  EXPECT_EQ(session.position(), 4);
  EXPECT_EQ(handle.metrics().n_prompt_tokens, 4);
  EXPECT_EQ(handle.metrics().n_prefilled_tokens, 4);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_EQ(executor.wrap_calls.load(), 0);
  const auto slices = executor.slices();
  ASSERT_EQ(slices.size(), 2u);
  EXPECT_EQ(slices[0].base, -3);
  EXPECT_EQ(slices[0].offset, 3u);
  EXPECT_EQ(slices[0].size, 2u);
  EXPECT_EQ(slices[0].start(), 0);
  EXPECT_EQ(slices[0].values, (std::vector<std::int64_t>{-8, -9}));
  EXPECT_FALSE(slices[0].produce_output);
  EXPECT_EQ(slices[1].base, -3);
  EXPECT_EQ(slices[1].offset, 5u);
  EXPECT_EQ(slices[1].size, 2u);
  EXPECT_EQ(slices[1].start(), 2);
  EXPECT_EQ(slices[1].values, (std::vector<std::int64_t>{20, 21}));
  EXPECT_TRUE(slices[1].produce_output);
  for (const auto& slice : slices) {
    EXPECT_EQ(slice.identity, prepared.prepared.get());
    EXPECT_FALSE(slice.feedback);
  }
}

TEST(PreparationTest, PendingPredictionIsSeparateBeforeSelectedSuffix) {
  PrivateExecutor executor;
  GenerationCompletion first;
  GenerationCompletion second;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto initial = session.generate_async(
      std::vector<Token>{1},
      generation_config(),
      first.callback(),
      first.settled());
  ASSERT_TRUE(first.wait());
  ASSERT_EQ(initial.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_EQ(session.position(), 1);
  auto prepared = prepare(harness.runner, image_input())->take();
  ASSERT_TRUE(prepared.ok);
  auto handle = session.generate_async(
      prepared.prepared,
      3,
      5,
      generation_config(),
      second.callback(),
      second.settled());
  ASSERT_TRUE(second.wait());
  harness.runner.shutdown();

  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(second.tokens, (std::vector<Token>{1001}));
  EXPECT_EQ(second.terminal_calls, 1);
  EXPECT_EQ(session.position(), 7);
  EXPECT_EQ(handle.metrics().n_prompt_tokens, 5);
  EXPECT_EQ(handle.metrics().n_prefilled_tokens, 6);
  EXPECT_EQ(executor.prepare_calls.load(), 2);
  EXPECT_EQ(executor.wrap_calls.load(), 1);
  const auto slices = executor.slices();
  ASSERT_EQ(slices.size(), 5u);
  EXPECT_TRUE(slices[1].feedback);
  EXPECT_NE(slices[1].identity, prepared.prepared.get());
  EXPECT_EQ(slices[1].values, (std::vector<std::int64_t>{1000}));
  EXPECT_EQ(slices[1].offset, 0u);
  EXPECT_EQ(slices[1].size, 1u);
  EXPECT_EQ(slices[1].start(), 1);
  EXPECT_FALSE(slices[1].produce_output);
  for (std::size_t i = 2; i < slices.size(); ++i) {
    EXPECT_EQ(slices[i].identity, prepared.prepared.get());
    EXPECT_EQ(slices[i].base, -1);
    EXPECT_EQ(slices[i].offset, 3u + 2u * (i - 2));
    EXPECT_EQ(slices[i].start(), 2 + 2 * static_cast<std::int64_t>(i - 2));
    EXPECT_EQ(slices[i].size, i == 4 ? 1u : 2u);
    EXPECT_EQ(slices[i].produce_output, i == 4);
    EXPECT_FALSE(slices[i].feedback);
  }
  EXPECT_EQ(slices.back().values, (std::vector<std::int64_t>{22}));
}

TEST(PreparationTest, TokenOverloadPreparesOnceAndWrapsDecodeOnEngineThread) {
  PrivateExecutor executor;
  GenerationCompletion completion;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto handle = session.generate_async(
      std::vector<Token>{11, 12, 13, 14},
      generation_config(3),
      completion.callback(),
      completion.settled());
  ASSERT_TRUE(completion.wait());
  harness.runner.shutdown();

  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(completion.tokens, (std::vector<Token>{1000, 1001, 1002}));
  EXPECT_EQ(completion.terminal_calls, 1);
  EXPECT_EQ(completion.settled_calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_EQ(executor.wrap_calls.load(), 2);
  EXPECT_EQ(session.position(), 6);
  const auto slices = executor.slices();
  ASSERT_EQ(slices.size(), 4u);
  EXPECT_FALSE(slices[0].feedback);
  EXPECT_FALSE(slices[0].produce_output);
  EXPECT_FALSE(slices[1].feedback);
  EXPECT_TRUE(slices[1].produce_output);
  for (std::size_t i = 2; i < slices.size(); ++i) {
    EXPECT_TRUE(slices[i].feedback);
    EXPECT_TRUE(slices[i].produce_output);
    EXPECT_EQ(slices[i].offset, 0u);
    EXPECT_EQ(slices[i].size, 1u);
    EXPECT_EQ(slices[i].start(), static_cast<std::int64_t>(i + 2));
    EXPECT_EQ(
        slices[i].values,
        (std::vector<std::int64_t>{998 + static_cast<std::int64_t>(i)}));
  }
  const auto threads = executor.threads();
  ASSERT_FALSE(threads.empty());
  EXPECT_NE(threads.front(), std::this_thread::get_id());
  for (auto thread : threads) {
    EXPECT_EQ(thread, threads.front());
  }
}

TEST(PreparationTest, QueuedCancellationCompletesOnceWithoutPreparing) {
  PrivateExecutor executor(test_preparation_config(1));
  executor.initialize_gate.hold();
  Harness harness(executor);
  EXPECT_TRUE(executor.initialize_gate.wait_for_entry());
  auto cancellation = std::make_shared<std::atomic<bool>>(false);
  auto cancelled = prepare(harness.runner, image_input(), cancellation);
  cancellation->store(true);
  executor.initialize_gate.release();
  auto result = cancelled->take();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  EXPECT_EQ(executor.prepare_calls.load(), 0);

  auto recovered = prepare(harness.runner)->take();
  EXPECT_TRUE(recovered.ok) << "cancellation must release its reservation";
  harness.runner.shutdown();
  EXPECT_EQ(cancelled->calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
}

TEST(
    PreparationTest,
    InFlightCancellationDiscardsOutputAndReleasesReservation) {
  PrivateExecutor executor(test_preparation_config(1));
  executor.preparation_gate.hold();
  Harness harness(executor);
  auto cancellation = std::make_shared<std::atomic<bool>>(false);
  auto cancelled = prepare(harness.runner, image_input(), cancellation);
  EXPECT_TRUE(executor.preparation_gate.wait_for_entry());
  cancellation->store(true);
  executor.preparation_gate.release();
  auto result = cancelled->take();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  const auto created = executor.created();
  ASSERT_EQ(created.size(), 1u);
  EXPECT_TRUE(created.front().expired());

  auto recovered = prepare(harness.runner)->take();
  EXPECT_TRUE(recovered.ok);
  harness.runner.shutdown();
  EXPECT_EQ(cancelled->calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

TEST(PreparationTest, ShutdownCompletesQueuedAndNewPreparationExactlyOnce) {
  PrivateExecutor executor;
  auto queued = std::make_shared<PreparationCompletion>();
  auto capture = std::make_shared<int>(42);
  std::weak_ptr<int> retained_capture = capture;
  GenerationCompletion generation;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto handle = session.generate_async(
      std::vector<Token>{1},
      generation_config(),
      [&, capture = std::move(capture)](const GenerationUpdate& update) {
        generation.callback()(update);
        if (update.finish_reason) {
          harness.runner.prepare_async(
              image_input(),
              {},
              [queued, capture](bool ok, PreparedInputPtr output) {
                queued->complete(ok, std::move(output));
              });
          harness.runner.shutdown();
        }
      },
      generation.settled());
  ASSERT_TRUE(generation.wait());
  harness.runner.shutdown();
  auto result = queued->take();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  EXPECT_EQ(queued->calls.load(), 1);
  EXPECT_TRUE(retained_capture.expired());
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);

  auto rejected = prepare(harness.runner);
  auto rejection = rejected->take();
  EXPECT_FALSE(rejection.ok);
  EXPECT_EQ(rejection.prepared, nullptr);
  EXPECT_EQ(rejection.thread, std::this_thread::get_id());
  harness.runner.shutdown();
  EXPECT_EQ(rejected->calls.load(), 1);
  EXPECT_EQ(queued->calls.load(), 1);
}

class InvalidPreparationTest
    : public ::testing::TestWithParam<PreparationReply> {};

TEST_P(InvalidPreparationTest, FailsOnceReleasesOutputAndAllowsRetry) {
  PrivateExecutor executor(test_preparation_config(1));
  executor.next_reply.store(GetParam());
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto failed = prepare(harness.runner);
  auto result = failed->take();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  EXPECT_EQ(session.position(), 0);
  EXPECT_TRUE(executor.slices().empty());
  for (const auto& created : executor.created()) {
    EXPECT_TRUE(created.expired());
  }

  auto retry = prepare(harness.runner);
  auto recovered = retry->take();
  EXPECT_TRUE(recovered.ok) << "invalid output must return the reserved budget";
  harness.runner.shutdown();
  EXPECT_EQ(failed->calls.load(), 1);
  EXPECT_EQ(retry->calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

INSTANTIATE_TEST_SUITE_P(
    PreparationTest,
    InvalidPreparationTest,
    ::testing::Values(
        PreparationReply::Null,
        PreparationReply::Empty,
        PreparationReply::ForeignTag,
        PreparationReply::TooManyPositions,
        PreparationReply::TooManyBytes,
#if ET_HAS_EXCEPTIONS
        PreparationReply::ExceptionWithOutput,
#endif
        PreparationReply::FailureWithOutput));

TEST(PreparationTest, AggregateBudgetReservesMaximumBeforePreparation) {
  auto config = test_preparation_config(2);
  config.max_total_retained_bytes = 2 *
      (config.max_retained_bytes + *config.input_retained_bytes(text_input()));
  PrivateExecutor executor(config);
  executor.preparation_gate.hold();
  Harness harness(executor);
  auto first = prepare(harness.runner);
  EXPECT_TRUE(executor.preparation_gate.wait_for_entry());
  auto second = prepare(harness.runner);
  auto rejected = prepare(harness.runner);
  // Both reservations exist, although only one executor call has begun.
  auto rejection = rejected->take();
  EXPECT_FALSE(rejection.ok);
  EXPECT_EQ(rejection.prepared, nullptr);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  executor.preparation_gate.release();
  auto a = first->take();
  auto b = second->take();
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  ASSERT_LT(
      a.prepared->retained_bytes(),
      harness.runner.preparation_config().max_retained_bytes);
  auto compacted = prepare(harness.runner)->take();
  EXPECT_TRUE(compacted.ok)
      << "successful owners retain their actual byte charge";
  auto still_full = prepare(harness.runner)->take();
  EXPECT_FALSE(still_full.ok)
      << "actual owners plus the next reservation exceed capacity";

  a.prepared.reset();
  auto admitted = prepare(harness.runner)->take();
  EXPECT_TRUE(admitted.ok);
  harness.runner.shutdown();
  EXPECT_EQ(first->calls.load(), 1);
  EXPECT_EQ(second->calls.load(), 1);
  EXPECT_EQ(rejected->calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 4);
}

TEST(PreparationTest, ReservationFollowsAliasesAndInFlightExecutionOwners) {
  PrivateExecutor executor(test_preparation_config(1));
  GenerationCompletion generation;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto prepared = prepare(harness.runner, image_input())->take();
  ASSERT_TRUE(prepared.ok);
  PreparedInputPtr alias(prepared.prepared, prepared.prepared.get());
  std::weak_ptr<const PreparedInput> owner = alias;
  executor.execution_gate.hold();
  auto handle = session.generate_async(
      prepared.prepared,
      3,
      4,
      generation_config(),
      generation.callback(),
      generation.settled());
  EXPECT_TRUE(executor.execution_gate.wait_for_entry());
  prepared.prepared.reset();
  alias.reset();
  EXPECT_FALSE(owner.expired());
  auto blocked = prepare(harness.runner)->take();
  EXPECT_FALSE(blocked.ok) << "executing slices still own the reservation";
  executor.execution_gate.release();
  ASSERT_TRUE(generation.wait());
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  // A command boundary ensures the final batch and its local owners are gone.
  auto barrier = open_session(harness.runner);
  ASSERT_TRUE(barrier.valid());
  EXPECT_TRUE(owner.expired());
  auto recovered = prepare(harness.runner)->take();
  EXPECT_TRUE(recovered.ok);
  harness.runner.shutdown();
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

TEST(PreparationTest, CallerAliasKeepsReservationAfterGenerationSettles) {
  PrivateExecutor executor(test_preparation_config(1));
  GenerationCompletion generation;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto prepared = prepare(harness.runner)->take();
  ASSERT_TRUE(prepared.ok);
  PreparedInputPtr alias(prepared.prepared, prepared.prepared.get());
  auto handle = session.generate_async(
      prepared.prepared,
      0,
      2,
      generation_config(),
      generation.callback(),
      generation.settled());
  prepared.prepared.reset();
  ASSERT_TRUE(generation.wait());
  auto barrier = open_session(harness.runner);
  ASSERT_TRUE(barrier.valid());
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  auto blocked = prepare(harness.runner)->take();
  EXPECT_FALSE(blocked.ok);
  alias.reset();
  auto recovered = prepare(harness.runner)->take();
  EXPECT_TRUE(recovered.ok);
  harness.runner.shutdown();
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

TEST(PreparationTest, PreparedOwnerCanOutliveRunnerAndExecutor) {
  PreparedInputPtr owner;
  std::weak_ptr<const PreparedInput> payload;
  {
    PrivateExecutor executor(test_preparation_config(1));
    Harness harness(executor);
    auto prepared = prepare(harness.runner, image_input())->take();
    ASSERT_TRUE(prepared.ok);
    owner = std::move(prepared.prepared);
    const auto created = executor.created();
    ASSERT_EQ(created.size(), 1u);
    payload = created.front();
  }
  EXPECT_FALSE(payload.expired());
  EXPECT_EQ(owner->position_count(), 10u);
  owner.reset();
  EXPECT_TRUE(payload.expired());
}

TEST(
    PreparationTest,
    BatchValidationAllowsNegativeBaseOnlyForValidMappedRange) {
  PrivateExecutor executor;
  PreparedInputPtr prepared;
  ASSERT_TRUE(executor.prepare(image_input(), prepared));
  BatchInput batch;
  batch.inputs.push_back(Input{1, false, 3, 4, prepared, -3});
  EXPECT_TRUE(executor.validate_batch(batch));
  batch.inputs.front().position = -4;
  EXPECT_FALSE(executor.validate_batch(batch));
  batch.inputs.front().position = -2;
  EXPECT_TRUE(executor.validate_batch(batch));
  batch.inputs.front().position = std::numeric_limits<Position>::max() - 4;
  EXPECT_FALSE(executor.validate_batch(batch));
  batch.inputs.front().position = std::numeric_limits<Position>::max();
  EXPECT_FALSE(executor.validate_batch(batch));
  batch.inputs.front().position = -3;
  batch.inputs.front().size = 8;
  EXPECT_FALSE(executor.validate_batch(batch));
  batch.inputs.front().size = 0;
  EXPECT_FALSE(executor.validate_batch(batch));
  batch.inputs.front().size = 1;
  batch.inputs.front().offset = std::numeric_limits<std::size_t>::max();
  EXPECT_FALSE(executor.validate_batch(batch));
}

TEST(
    PreparationTest,
    TokenHandleCancellationAndCloseDuringPreparationSettleOnce) {
  for (bool close : {false, true}) {
    PrivateExecutor executor;
    GenerationCompletion completion;
    Harness harness(executor);
    auto session = open_session(harness.runner);
    executor.preparation_gate.hold();
    auto handle = session.generate_async(
        std::vector<Token>{1, 2},
        generation_config(),
        completion.callback(),
        completion.settled());
    ASSERT_TRUE(executor.preparation_gate.wait_for_entry());
    if (close) {
      session = Session{};
    } else {
      handle.cancel();
    }
    executor.preparation_gate.release();
    ASSERT_TRUE(completion.wait());
    harness.runner.shutdown();
    EXPECT_EQ(handle.finish_reason(), FinishReason::Cancelled);
    EXPECT_EQ(completion.terminal_calls, 1);
    EXPECT_EQ(completion.settled_calls.load(), 1);
    EXPECT_TRUE(completion.tokens.empty());
    EXPECT_TRUE(executor.slices().empty());
    EXPECT_EQ(executor.prepare_calls.load(), 1);
    for (const auto& created : executor.created()) {
      EXPECT_TRUE(created.expired());
    }
  }
}

TEST(PreparationTest, ShutdownInsidePreparationDiscardsInflightOutput) {
  PrivateExecutor executor;
  Harness harness(executor);
  executor.after_prepare = [&] { harness.runner.shutdown(); };
  auto completion = prepare(harness.runner, image_input());
  auto result = completion->take();
  harness.runner.shutdown();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  EXPECT_EQ(completion->calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_TRUE(executor.slices().empty());
  for (const auto& created : executor.created()) {
    EXPECT_TRUE(created.expired());
  }
}

TEST(PreparationTest, FailedFeedbackDoesNotRunAnotherPreparationOrForward) {
  PrivateExecutor executor;
  GenerationCompletion completion;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  executor.fail_wrap = true;
  auto handle = session.generate_async(
      std::vector<Token>{1, 2},
      generation_config(2),
      completion.callback(),
      completion.settled());
  ASSERT_TRUE(completion.wait());
  harness.runner.shutdown();
  EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(completion.tokens, (std::vector<Token>{1000}));
  EXPECT_EQ(completion.terminal_calls, 1);
  EXPECT_EQ(completion.settled_calls.load(), 1);
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_EQ(executor.wrap_calls.load(), 1);
  EXPECT_EQ(executor.slices().size(), 1u);
  EXPECT_EQ(session.position(), 2);
}

TEST(
    PreparationTest,
    SourceContainerCapacityAndImageCountAreBoundedBeforeWork) {
  PrivateExecutor executor;
  Harness harness(executor);
  auto input = text_input({1});
  input.segments.reserve(
      executor.preparation_config().max_workspace_bytes /
          sizeof(MultimodalInput) +
      1);
  auto oversized = prepare(harness.runner, std::move(input))->take();
  EXPECT_FALSE(oversized.ok);
  auto images = image_input();
  images.segments.emplace_back(Image(std::vector<uint8_t>{8}, 1, 1, 1));
  auto excess_images = prepare(harness.runner, std::move(images))->take();
  EXPECT_FALSE(excess_images.ok);
  EXPECT_EQ(executor.prepare_calls.load(), 0);
  EXPECT_TRUE(prepare(harness.runner)->take().ok);
}

TEST(PreparationTest, WholeBatchRejectsForeignBackingBeforeAnyMutation) {
  PrivateExecutor executor;
  PreparedInputPtr accepted;
  PreparedInputPtr foreign;
  ASSERT_TRUE(executor.prepare(text_input(), accepted));
  executor.next_reply = PreparationReply::ForeignTag;
  ASSERT_TRUE(executor.prepare(text_input(), foreign));
  const auto left = executor.open_session();
  const auto right = executor.open_session();
  ASSERT_TRUE(left && right);
  BatchInput batch{
      {Input{*left, true, 0, 2, accepted, 0},
       Input{*right, true, 0, 2, foreign, 0}}};
  BatchOutput output;
  EXPECT_FALSE(executor.execute(batch, output));
  EXPECT_TRUE(executor.slices().empty());
  batch.inputs[1].prepared = accepted;
  EXPECT_TRUE(executor.execute(batch, output));
  EXPECT_EQ(executor.slices().size(), 2u);
}

TEST(
    PreparationTest,
    ShutdownPreparationAndConfigDoNotBorrowDestroyedExecutor) {
  auto executor = std::make_unique<PrivateExecutor>();
  Runner runner(*executor, DecodeFirstScheduler::create(3, 1, 2));
  runner.shutdown();
  executor.reset();
  EXPECT_EQ(runner.preparation_config().max_positions, 32u);
  auto completion = prepare(runner);
  auto result = completion->take();
  EXPECT_FALSE(result.ok);
  EXPECT_EQ(result.prepared, nullptr);
  EXPECT_EQ(completion->calls.load(), 1);
  EXPECT_EQ(result.thread, std::this_thread::get_id());
}

TEST(
    PreparationTest,
    TokenGenerationReusesItsSinglePreparationSlotForFeedback) {
  PrivateExecutor executor(test_preparation_config(1));
  GenerationCompletion completion;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto handle = session.generate_async(
      std::vector<Token>{1},
      generation_config(3),
      [&](const GenerationUpdate& update) {
        completion.callback()(update);
        for (const auto& input : executor.created()) {
          EXPECT_TRUE(input.expired())
              << "consumed backing must precede feedback reuse";
        }
      },
      completion.settled());
  ASSERT_TRUE(completion.wait());
  harness.runner.shutdown();
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(completion.tokens, (std::vector<Token>{1000, 1001, 1002}));
  EXPECT_EQ(executor.prepare_calls.load(), 1);
  EXPECT_EQ(executor.wrap_calls.load(), 2);
}

TEST(
    PreparationTest,
    NewPreparationsCannotTakeAnAdmittedGenerationsFeedbackSlot) {
  for (std::size_t slots : {1u, 2u}) {
    PrivateExecutor executor(test_preparation_config(slots));
    GenerationCompletion completion;
    auto peer = std::make_shared<PreparationCompletion>();
    Harness harness(executor);
    auto session = open_session(harness.runner);
    bool submitted_peer = false;
    auto handle = session.generate_async(
        std::vector<Token>{1},
        generation_config(3),
        [&](const GenerationUpdate& update) {
          completion.callback()(update);
          if (!submitted_peer && !update.tokens.empty()) {
            submitted_peer = true;
            harness.runner.prepare_async(
                text_input(), {}, [peer](bool ok, PreparedInputPtr output) {
                  peer->complete(ok, std::move(output));
                });
          }
        },
        completion.settled());
    ASSERT_TRUE(completion.wait());
    auto result = peer->take();
    harness.runner.shutdown();
    EXPECT_EQ(result.ok, slots == 2);
    EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
    EXPECT_EQ(completion.tokens, (std::vector<Token>{1000, 1001, 1002}));
    EXPECT_EQ(executor.prepare_calls.load(), slots == 2 ? 2 : 1);
    EXPECT_EQ(executor.wrap_calls.load(), 2);
  }
}

TEST(
    PreparationTest,
    SharedPreparedRangeRefusesFeedbackCapacityBeforeMutation) {
  auto config = test_preparation_config(1);
  config.max_total_retained_bytes = config.max_retained_bytes + 512;
  PrivateExecutor executor(config);
  GenerationCompletion rejected;
  GenerationCompletion retry;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto prepared = prepare(harness.runner)->take();
  ASSERT_TRUE(prepared.ok);
  auto refused = session.generate_async(
      prepared.prepared,
      0,
      2,
      generation_config(2),
      rejected.callback(),
      rejected.settled());
  ASSERT_TRUE(rejected.wait());
  EXPECT_EQ(refused.finish_reason(), FinishReason::Failed);
  EXPECT_TRUE(rejected.tokens.empty());
  EXPECT_TRUE(executor.slices().empty());
  EXPECT_EQ(session.position(), 0);
  auto accepted = session.generate_async(
      prepared.prepared,
      0,
      2,
      generation_config(1),
      retry.callback(),
      retry.settled());
  ASSERT_TRUE(retry.wait());
  EXPECT_EQ(accepted.finish_reason(), FinishReason::NewTokenLimit);
}

TEST(PreparationTest, FeedbackCapacityIsReleasedBeforeTheSettlementCallback) {
  PrivateExecutor executor(test_preparation_config(1));
  GenerationCompletion completion;
  auto next = std::make_shared<PreparationCompletion>();
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto handle = session.generate_async(
      std::vector<Token>{1}, generation_config(2), completion.callback(), [&] {
        harness.runner.prepare_async(
            text_input(), {}, [next](bool ok, PreparedInputPtr output) {
              next->complete(ok, std::move(output));
            });
        completion.settled()();
      });
  ASSERT_TRUE(completion.wait());
  auto result = next->take();
  harness.runner.shutdown();
  EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_TRUE(result.ok);
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

TEST(PreparationTest, QueuedSourceBytesShareTheAggregateOutputBudget) {
  auto config = test_preparation_config(2);
  config.max_workspace_bytes = 16 * 1024;
  auto source = text_input({1});
  std::vector<Token> tokens{1};
  tokens.reserve(1024);
  source.segments.clear();
  source.segments.emplace_back(std::move(tokens));
  const auto bytes = config.input_retained_bytes(source);
  ASSERT_TRUE(bytes);
  ASSERT_GT(*bytes, config.max_retained_bytes);
  config.max_total_retained_bytes = 2 * (*bytes + config.max_retained_bytes);
  PrivateExecutor executor(config);
  executor.initialize_gate.hold();
  Harness harness(executor);
  ASSERT_TRUE(executor.initialize_gate.wait_for_entry());
  // Copying vectors may reduce capacity; explicitly build each owned source.
  auto make_source = [] {
    PreparationInput input;
    std::vector<Token> tokens{1};
    tokens.reserve(1024);
    input.segments.emplace_back(std::move(tokens));
    return input;
  };
  auto first = prepare(harness.runner, make_source());
  auto second = prepare(harness.runner, make_source());
  auto third = prepare(harness.runner, make_source());
  EXPECT_EQ(
      third->future.wait_for(std::chrono::seconds(0)),
      std::future_status::ready);
  executor.initialize_gate.release();
  EXPECT_FALSE(third->take().ok);
  EXPECT_TRUE(first->take().ok);
  EXPECT_TRUE(second->take().ok);
  harness.runner.shutdown();
  EXPECT_EQ(executor.prepare_calls.load(), 2);
}

TEST(PreparationTest, ExecutorRetainedInputCannotDoubleSpendFeedbackCapacity) {
  PrivateExecutor executor(test_preparation_config(1));
  executor.retain_inputs = true;
  GenerationCompletion completion;
  Harness harness(executor);
  auto session = open_session(harness.runner);
  auto handle = session.generate_async(
      std::vector<Token>{1},
      generation_config(2),
      completion.callback(),
      completion.settled());
  ASSERT_TRUE(completion.wait());
  EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(executor.wrap_calls.load(), 0);
  EXPECT_FALSE(prepare(harness.runner)->take().ok)
      << "the retained input still owns the original quota";
  executor.retained_inputs.clear();
  EXPECT_TRUE(prepare(harness.runner)->take().ok);
  harness.runner.shutdown();
}

TEST(PreparationTest, TokenCommandInboxReservesSourceBeforeEngineAdmission) {
  for (bool cancel_first : {false, true}) {
    auto config = test_preparation_config();
    config.max_workspace_bytes = 16 * 1024;
    config.max_total_retained_bytes = 20 * 1024;
    PrivateExecutor executor(config);
    GenerationCompletion blocker_completion;
    GenerationCompletion retry_completion;
    std::vector<std::unique_ptr<GenerationCompletion>> completions;
    Harness harness(executor);
    auto blocker_session = open_session(harness.runner);
    std::vector<Session> sessions;
    for (int i = 0; i < 10; ++i) {
      sessions.push_back(open_session(harness.runner));
    }
    executor.execution_gate.hold();
    auto blocker = blocker_session.generate_async(
        std::vector<Token>{1},
        generation_config(),
        blocker_completion.callback(),
        blocker_completion.settled());
    ASSERT_TRUE(executor.execution_gate.wait_for_entry());
    auto large_capacity_tokens = [] {
      std::vector<Token> tokens{1};
      tokens.reserve(1024);
      return tokens;
    };
    std::vector<GenerationHandle> handles;
    for (int i = 0; i < 10; ++i) {
      completions.push_back(std::make_unique<GenerationCompletion>());
      auto& completion = *completions.back();
      handles.push_back(sessions[i].generate_async(
          large_capacity_tokens(),
          generation_config(2),
          completion.callback(),
          completion.settled()));
      EXPECT_EQ(
          completion.future.wait_for(std::chrono::seconds(0)),
          i < 2 ? std::future_status::timeout : std::future_status::ready);
      if (i >= 2) {
        EXPECT_EQ(handles.back().finish_reason(), FinishReason::Failed);
      }
    }
    EXPECT_EQ(executor.prepare_calls.load(), 1)
        << "the engine has not drained any queued token request";
    if (cancel_first) {
      handles.front().cancel();
    }
    executor.execution_gate.release();
    ASSERT_TRUE(blocker_completion.wait());
    for (std::size_t i = 0; i < completions.size(); ++i) {
      ASSERT_TRUE(completions[i]->wait());
      EXPECT_EQ(completions[i]->terminal_calls, 1);
      EXPECT_EQ(completions[i]->settled_calls.load(), 1);
      if (i < 2) {
        EXPECT_EQ(
            handles[i].finish_reason(),
            cancel_first && i == 0 ? FinishReason::Cancelled
                                   : FinishReason::NewTokenLimit);
      }
    }
    EXPECT_EQ(executor.prepare_calls.load(), cancel_first ? 2 : 3);
    EXPECT_EQ(executor.wrap_calls.load(), cancel_first ? 1 : 2);
    auto retry = sessions.back().generate_async(
        large_capacity_tokens(),
        generation_config(),
        retry_completion.callback(),
        retry_completion.settled());
    ASSERT_TRUE(retry_completion.wait());
    EXPECT_EQ(retry.finish_reason(), FinishReason::NewTokenLimit);
  }
}

TEST(PreparationTest, ExternalPreparedInboxOwnsInputAndFeedbackReservations) {
  for (bool cancel_first : {false, true}) {
    auto config = test_preparation_config();
    config.max_workspace_bytes = 16 * 1024;
    config.max_total_retained_bytes = 6 * config.max_retained_bytes;
    PrivateExecutor executor(config);
    std::vector<PreparedInputPtr> external;
    for (int i = 0; i < 10; ++i) {
      PreparedInputPtr input;
      ASSERT_TRUE(executor.prepare(text_input({1}), input));
      external.push_back(std::move(input));
    }
    const auto owners = executor.created();
    GenerationCompletion blocker_completion;
    std::vector<std::unique_ptr<GenerationCompletion>> completions;
    Harness harness(executor);
    auto blocker_session = open_session(harness.runner);
    std::vector<Session> sessions;
    for (int i = 0; i < 10; ++i) {
      sessions.push_back(open_session(harness.runner));
    }
    executor.execution_gate.hold();
    auto blocker = blocker_session.generate_async(
        std::vector<Token>{1},
        generation_config(),
        blocker_completion.callback(),
        blocker_completion.settled());
    ASSERT_TRUE(executor.execution_gate.wait_for_entry());
    std::vector<GenerationHandle> handles;
    for (int i = 0; i < 10; ++i) {
      completions.push_back(std::make_unique<GenerationCompletion>());
      auto& completion = *completions.back();
      handles.push_back(sessions[i].generate_async(
          std::move(external[i]),
          0,
          1,
          generation_config(2),
          completion.callback(),
          completion.settled()));
      EXPECT_EQ(
          completion.future.wait_for(std::chrono::seconds(0)),
          i < 2 ? std::future_status::timeout : std::future_status::ready);
      if (i >= 2) {
        EXPECT_EQ(handles.back().finish_reason(), FinishReason::Failed);
        EXPECT_TRUE(owners[i].expired());
      }
    }
    EXPECT_EQ(executor.prepare_calls.load(), 11);
    if (cancel_first) {
      handles.front().cancel();
    }
    executor.execution_gate.release();
    ASSERT_TRUE(blocker_completion.wait());
    for (std::size_t i = 0; i < completions.size(); ++i) {
      ASSERT_TRUE(completions[i]->wait());
      EXPECT_EQ(completions[i]->terminal_calls, 1);
      EXPECT_EQ(completions[i]->settled_calls.load(), 1);
      EXPECT_TRUE(owners[i].expired());
      if (i < 2) {
        EXPECT_EQ(
            handles[i].finish_reason(),
            cancel_first && i == 0 ? FinishReason::Cancelled
                                   : FinishReason::NewTokenLimit);
      }
    }
    EXPECT_EQ(executor.prepare_calls.load(), 11);
    EXPECT_EQ(executor.wrap_calls.load(), cancel_first ? 1 : 2);
    // Rejected requests could reserve input but not feedback. A leaked input
    // reservation would leave too little capacity for this large source.
    std::vector<Token> tokens{1};
    tokens.reserve(512);
    EXPECT_TRUE(
        prepare(harness.runner, text_input(std::move(tokens)))->take().ok);
  }
}

} // namespace
} // namespace executorch::extension::llm::batching

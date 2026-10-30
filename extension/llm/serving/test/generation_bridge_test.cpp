/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/detail/generation_bridge.h>
#include <executorch/extension/llm/serving/serving_runtime.h>
#include <executorch/extension/llm/serving/test/prepared_input.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <future>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <type_traits>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using namespace executorch::extension::llm;
using serving::ErrorCode;
using serving::RequestHandle;
using serving::ServingError;
using serving::ServingRuntime;
using serving::ServingRuntimeConfig;
using serving::detail::GenerationBridge;
using serving::detail::GenerationCompletion;
using serving::detail::GenerationRequest;
using serving::detail::SubmissionResult;
using serving::testing::TestPreparedInput;

namespace {

constexpr std::chrono::seconds kTimeout{5};

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
  bool wait_for(
      std::size_t count = 1,
      std::chrono::seconds timeout = kTimeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return arrivals_ >= count; });
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

class Executor : public batching::testing::FakeExecutor {
 public:
  bool initialize() override {
    engine_thread = std::this_thread::get_id();
    initializing.arrive();
    return FakeExecutor::initialize();
  }
  std::optional<batching::SessionId> open_session() override {
    opening.arrive();
    return FakeExecutor::open_session();
  }
  std::optional<batching::SessionId> clone(
      batching::SessionId,
      batching::Position) override {
    ++clones;
    return std::nullopt;
  }
  bool execute(const batching::BatchInput& input, batching::BatchOutput& output)
      override {
    if (on_execute) {
      on_execute(input);
    }
    std::set<batching::SessionId> sessions;
    for (const auto& item : input.inputs) {
      sessions.insert(item.sid);
    }
    const auto call = ++calls;
    if (sessions.size() >= 2) {
      shared_batch.store(true);
    }
    executing.arrive();
    if (call >= 2) {
      subsequent.arrive();
    }
    return FakeExecutor::execute(input, output);
  }
  void release_all() {
    initializing.release();
    opening.release();
    executing.release();
    subsequent.release();
  }
  std::thread::id engine_thread;
  std::function<void(const batching::BatchInput&)> on_execute;
  Gate initializing;
  Gate opening;
  Gate executing;
  Gate subsequent;
  std::atomic<int> calls{0};
  std::atomic<int> clones{0};
  std::atomic<bool> shared_batch{false};
};

class PreparedExecutor : public Executor {
 public:
  bool accepts(const batching::PreparedInput& input) const override {
    EXPECT_EQ(std::this_thread::get_id(), engine_thread);
    return input.kind() == &TestPreparedInput::tag;
  }
  bool execute(const batching::BatchInput& input, batching::BatchOutput& output)
      override {
    for (const auto& slice : input.inputs) {
      if (const auto* prepared =
              std::get_if<batching::PreparedInputPtr>(&slice.payload)) {
        if (!*prepared || (*prepared)->kind() != &TestPreparedInput::tag ||
            slice.offset > (*prepared)->size() ||
            slice.size > (*prepared)->size() - slice.offset) {
          return false;
        }
      }
      slices.push_back(slice);
    }
    // This size-only model generates from sampling state, not payload values.
    return Executor::execute(input, output);
  }
  std::vector<batching::Input> slices;
};

struct Events {
  void record(const batching::GenerationUpdate& update) {
    std::lock_guard<std::mutex> lock(mutex);
    updates.push_back(update);
    tokens_seen += update.tokens.size();
    sink_thread = std::this_thread::get_id();
    sink_threads.insert(sink_thread);
    if (update.finish_reason) {
      ++terminals;
      terminal_saw_commit = completion.has_value();
    }
  }
  std::mutex mutex;
  std::vector<batching::GenerationUpdate> updates;
  std::optional<GenerationCompletion> completion;
  int terminals = 0;
  bool terminal_saw_commit = false;
  bool prepared = false;
  bool commit_saw_preparation = false;
  std::size_t tokens_seen = 0;
  std::size_t tokens_seen_at_prepare = 0;
  std::thread::id sink_thread;
  std::set<std::thread::id> sink_threads;
  std::thread::id prepare_thread;
  std::thread::id commit_thread;
};

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

RequestHandle accepted(SubmissionResult result) {
  if (auto* error = std::get_if<ServingError>(&result)) {
    ADD_FAILURE() << error->message;
    return {};
  }
  return std::get<RequestHandle>(std::move(result));
}

void completed(const RequestHandle& handle) {
  ASSERT_NE(handle.id(), 0u);
  ASSERT_TRUE(wait_until([&] { return handle.done(); }));
  handle.wait();
}

class GenerationBridgeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
  void watch_callbacks() {
    // Reentry regressions can deadlock inside submit/info/done or teardown.
    callback_watchdog = std::thread([this] {
      if (!test_finished.wait_for(1, kTimeout * 6)) {
        ADD_FAILURE() << "callback test or teardown deadlocked";
        std::abort();
      }
    });
  }
  void start(ServingRuntimeConfig config = {4, 128, 16}) {
    runtime = std::make_unique<ServingRuntime>(
        executor, batching::DecodeFirstScheduler::create(32, 4, 8), config);
  }
  void opened(const std::string& key) {
    auto future = runtime->open_session_async(key);
    ASSERT_EQ(future.wait_for(kTimeout), std::future_status::ready);
    EXPECT_FALSE(future.get());
  }
  GenerationRequest request(
      std::optional<std::string> key,
      const std::shared_ptr<Events>& events,
      int budget = 2) {
    GenerationRequest result;
    result.key = std::move(key);
    result.delta = std::vector<batching::Token>{10, 11};
    result.config.max_new_tokens = budget;
    result.config.seed = 42;
    result.on_update = [events](
                           const batching::GenerationUpdate& update,
                           const RequestHandle&) { events->record(update); };
    result.on_prepare_complete =
        [events](const GenerationCompletion& completion) {
          std::lock_guard<std::mutex> lock(events->mutex);
          events->prepared = true;
          EXPECT_TRUE(completion.terminal.tokens.empty());
          events->tokens_seen_at_prepare = events->tokens_seen;
          events->prepare_thread = std::this_thread::get_id();
        };
    result.on_complete = [events](const GenerationCompletion& completion) {
      std::lock_guard<std::mutex> lock(events->mutex);
      events->completion = completion;
      events->commit_saw_preparation = events->prepared;
      events->commit_thread = std::this_thread::get_id();
    };
    return result;
  }
  void expect_terminal(
      const std::shared_ptr<Events>& events,
      batching::FinishReason reason) {
    std::lock_guard<std::mutex> lock(events->mutex);
    ASSERT_EQ(events->terminals, 1);
    ASSERT_FALSE(events->updates.empty());
    EXPECT_EQ(events->updates.back().finish_reason, reason);
    EXPECT_TRUE(events->terminal_saw_commit);
    EXPECT_TRUE(events->commit_saw_preparation);
    EXPECT_NE(events->sink_thread, events->commit_thread);
    EXPECT_NE(events->sink_thread, executor.engine_thread);
    EXPECT_NE(events->prepare_thread, events->commit_thread);
    EXPECT_NE(events->prepare_thread, executor.engine_thread);
  }
  void TearDown() override {
    executor.release_all();
    sink_gate.release();
    other_sink_gate.release();
    cleanup_gate.release();
    for (auto& caller : callback_callers) {
      if (caller.joinable()) {
        caller.join();
      }
    }
    if (runtime) {
      runtime->shutdown();
    }
    runtime.reset();
    EXPECT_EQ(executor.clones.load(), 0);
    test_finished.arrive();
    if (callback_watchdog.joinable()) {
      callback_watchdog.join();
    }
  }
  Executor executor;
  Gate sink_gate;
  Gate other_sink_gate;
  Gate cleanup_gate;
  struct SmallCallbackState {
    std::atomic<int> live_owners{0};
    std::atomic<int> destructions{0};
    std::atomic<int> calls{0};
    std::atomic<int> terminals{0};
  } small_callback_state;
  RequestHandle cleanup_handle;
  std::vector<std::thread> callback_callers;
  Gate test_finished;
  std::thread callback_watchdog;
  std::unique_ptr<ServingRuntime> runtime;
};

class GenerationFenceTest : public GenerationBridgeTest,
                            public ::testing::WithParamInterface<bool> {
 protected:
  void fence(std::string key, serving::LifecycleCallback callback) {
    if (GetParam()) {
      runtime->reset_session_async(std::move(key), std::move(callback));
    } else {
      runtime->close_session_async(std::move(key), std::move(callback));
    }
  }

  std::future<serving::LifecycleResult> fence(std::string key) {
    return GetParam() ? runtime->reset_session_async(std::move(key))
                      : runtime->close_session_async(std::move(key));
  }
};

INSTANTIATE_TEST_SUITE_P(
    CloseAndReset,
    GenerationFenceTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Reset" : "Close";
    });

enum class InlineCallbackSlot { Update, PrepareComplete, Complete };
enum class InlineCallbackBoundary { Wait, Close, Reset, Shutdown };
using InlineCallbackParam =
    std::tuple<InlineCallbackSlot, InlineCallbackBoundary>;

class GenerationInlineCallbackTest
    : public GenerationBridgeTest,
      public ::testing::WithParamInterface<InlineCallbackParam> {
 protected:
  struct InlineCallbackState {
    Gate* cleanup = nullptr;
    Gate* terminal_gate = nullptr;
    std::thread::id submitting_thread;
    std::atomic<int> live_owners{0};
    std::atomic<int> copies{0};
    std::atomic<int> original_destructions{0};
    std::atomic<int> copied_destructions{0};
    std::atomic<int> terminals{0};
    std::atomic<int> completion_calls{0};
    std::atomic<bool> cleanup_entered{false};
    std::promise<RequestHandle> terminal_handle;

    void update(
        const batching::GenerationUpdate& update,
        const RequestHandle& handle) {
      if (update.finish_reason) {
        EXPECT_EQ(update.finish_reason, batching::FinishReason::NewTokenLimit);
        EXPECT_TRUE(update.tokens.empty());
        ++terminals;
        terminal_handle.set_value(handle);
        if (terminal_gate) {
          terminal_gate->arrive();
        }
      }
    }
  } inline_callback_state;

  struct Probe {
    InlineCallbackState* state;
    int depth = 0;

    explicit Probe(InlineCallbackState* state) noexcept : state(state) {
      ++state->live_owners;
    }
    Probe(const Probe& other) noexcept
        : state(other.state), depth(other.depth + 1) {
      if (state) {
        ++state->live_owners;
        ++state->copies;
      }
    }
    Probe(Probe&& other) noexcept : state(other.state), depth(other.depth) {
      other.state = nullptr;
    }
    ~Probe() {
      if (!state) {
        return;
      }
      if (depth == 0) {
        ++state->original_destructions;
      } else {
        ++state->copied_destructions;
        if (state->cleanup &&
            std::this_thread::get_id() == state->submitting_thread) {
          state->cleanup_entered.store(true);
          state->cleanup->arrive();
        }
      }
      --state->live_owners;
    }
    void operator()(
        const batching::GenerationUpdate& update,
        const RequestHandle& handle) const {
      state->update(update, handle);
    }
    void operator()(const GenerationCompletion&) const noexcept {
      ++state->completion_calls;
    }
  };
  static_assert(sizeof(Probe) <= 2 * sizeof(void*));
  static_assert(std::is_nothrow_copy_constructible<Probe>::value);
  static_assert(std::is_nothrow_move_constructible<Probe>::value);

  static GenerationRequest inline_request(
      InlineCallbackSlot slot,
      InlineCallbackState& state) {
    using Update = decltype(GenerationRequest::on_update);
    using Completion = decltype(GenerationRequest::on_complete);
    batching::GenConfig config;
    config.max_new_tokens = 1;
    config.seed = 42;
    // Prvalue construction leaves no caller-owned function/request source.
    return GenerationRequest{
        "session",
        std::vector<batching::Token>{10, 11},
        config,
        slot == InlineCallbackSlot::Update
            ? Update{Probe{&state}}
            : Update{[state = &state](
                         const batching::GenerationUpdate& update,
                         const RequestHandle& handle) {
                state->update(update, handle);
              }},
        slot == InlineCallbackSlot::PrepareComplete ? Completion{Probe{&state}}
                                                    : Completion{},
        slot == InlineCallbackSlot::Complete ? Completion{Probe{&state}}
                                             : Completion{}};
  }

  static bool supports_inline_staging(InlineCallbackSlot slot) {
    bool supported = true;
    {
      InlineCallbackState state;
      {
        auto incoming = inline_request(slot, state);
        supported &= state.live_owners.load() == 1 && state.copies.load() == 0;
        GenerationRequest parameter(std::move(incoming));
        auto published =
            std::make_shared<GenerationRequest>(std::move(parameter));
        // The old bridge -> Impl -> RequestState path must not destroy a
        // positive-depth copy before publication, where gating would deadlock.
        supported &= state.live_owners.load() == 3 &&
            state.copies.load() == 2 &&
            state.original_destructions.load() == 0 &&
            state.copied_destructions.load() == 0;
      }
      supported &= state.live_owners.load() == 0 &&
          state.original_destructions.load() == 1 &&
          state.copied_destructions.load() == 2;
    }
    {
      InlineCallbackState state;
      {
        auto incoming = inline_request(slot, state);
        auto published =
            std::make_shared<GenerationRequest>(std::move(incoming));
        incoming = {};
        // The fixed path resets only the depth-zero bridge parameter before
        // admission; the depth-one owner survives until dispatcher cleanup.
        supported &= !incoming.on_update && !incoming.on_prepare_complete &&
            !incoming.on_complete && state.live_owners.load() == 1 &&
            state.copies.load() == 1 &&
            state.original_destructions.load() == 1 &&
            state.copied_destructions.load() == 0;
      }
      supported &= state.live_owners.load() == 0 &&
          state.copied_destructions.load() == 1;
    }
    return supported;
  }
};

TEST_P(
    GenerationInlineCallbackTest,
    BoundaryIncludesSubmittingThreadInlineCallbackOwners) {
  const auto slot = std::get<0>(GetParam());
  const auto boundary = std::get<1>(GetParam());
  if (!supports_inline_staging(slot)) {
    GTEST_SKIP() << "Requires inline std::function cloning on move and "
                    "depth-zero-only source reset";
  }
  watch_callbacks();
  start();
  cleanup_gate.hold();
  sink_gate.hold();
  auto& state = inline_callback_state;
  state.cleanup = &cleanup_gate;
  state.terminal_gate = &sink_gate;
  auto terminal = state.terminal_handle.get_future();
  auto submitted = std::make_shared<std::promise<SubmissionResult>>();
  auto returned = submitted->get_future();
  callback_callers.emplace_back([this, slot, submitted] {
    inline_callback_state.submitting_thread = std::this_thread::get_id();
    // Finish the full expression, including parameter/temporary destruction,
    // before reporting return. Probe's original temporary was moved empty.
    auto result = GenerationBridge::submit(
        *runtime, inline_request(slot, inline_callback_state));
    submitted->set_value(std::move(result));
  });
  ASSERT_TRUE(wait_until([&] {
    return returned.wait_for(std::chrono::seconds(0)) ==
        std::future_status::ready ||
        state.cleanup_entered.load();
  }));
  ASSERT_EQ(terminal.wait_for(kTimeout), std::future_status::ready);
  const auto handle = terminal.get();
  ASSERT_NE(handle.id(), 0u);
  ASSERT_TRUE(sink_gate.wait_for());
  EXPECT_FALSE(handle.done());

  auto finished = std::make_shared<std::promise<int>>();
  auto owners_at_boundary = finished->get_future();
  callback_callers.emplace_back([this, boundary, handle, finished] {
    switch (boundary) {
      case InlineCallbackBoundary::Wait:
        handle.wait();
        break;
      case InlineCallbackBoundary::Close:
        EXPECT_FALSE(runtime->close_session_async("session").get());
        break;
      case InlineCallbackBoundary::Reset:
        EXPECT_FALSE(runtime->reset_session_async("session").get());
        break;
      case InlineCallbackBoundary::Shutdown:
        runtime->shutdown();
        break;
    }
    finished->set_value(inline_callback_state.live_owners.load());
  });
  sink_gate.release();
  ASSERT_EQ(owners_at_boundary.wait_for(kTimeout), std::future_status::ready);
  EXPECT_EQ(owners_at_boundary.get(), 0)
      << "Completion boundary preceded a submitting-thread inline owner";
  EXPECT_TRUE(handle.done());
  EXPECT_EQ(state.live_owners.load(), 0);
  EXPECT_EQ(state.terminals.load(), 1);
  EXPECT_EQ(
      state.completion_calls.load(),
      slot == InlineCallbackSlot::Update ? 0 : 1);
  EXPECT_FALSE(handle.error());

  // On the old path the Impl parameter is held at depth one, with its own
  // owner and the bridge parameter's owner still alive after this boundary.
  cleanup_gate.release();
  ASSERT_EQ(returned.wait_for(kTimeout), std::future_status::ready);
  auto result = returned.get();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  EXPECT_EQ(std::get<RequestHandle>(result).id(), handle.id());
  for (auto& caller : callback_callers) {
    caller.join();
  }
  EXPECT_EQ(state.live_owners.load(), 0);
}

INSTANTIATE_TEST_SUITE_P(
    AllSlotsAndBoundaries,
    GenerationInlineCallbackTest,
    ::testing::Combine(
        ::testing::Values(
            InlineCallbackSlot::Update,
            InlineCallbackSlot::PrepareComplete,
            InlineCallbackSlot::Complete),
        ::testing::Values(
            InlineCallbackBoundary::Wait,
            InlineCallbackBoundary::Close,
            InlineCallbackBoundary::Reset,
            InlineCallbackBoundary::Shutdown)),
    ([](const ::testing::TestParamInfo<InlineCallbackParam>& info) {
      const char* slots[] = {"Update", "PrepareComplete", "Complete"};
      const char* boundaries[] = {"Wait", "Close", "Reset", "Shutdown"};
      return std::string(slots[static_cast<int>(std::get<0>(info.param))]) +
          boundaries[static_cast<int>(std::get<1>(info.param))];
    }));

TEST_F(GenerationBridgeTest, SmallUpdateCallbackDestructionCanReenterInfo) {
  watch_callbacks();
  executor.executing.hold();
  ServingRuntimeConfig config{4, 128, 16};
  config.max_requests = 1;
  start(config);
  // Nothrow copying keeps this eligible for libc++'s inline function storage.
  struct SmallCallback {
    ServingRuntime* runtime;
    SmallCallbackState* state;

    SmallCallback(ServingRuntime* runtime, SmallCallbackState* state) noexcept
        : runtime(runtime), state(state) {
      ++state->live_owners;
    }
    SmallCallback(const SmallCallback& other) noexcept
        : SmallCallback(other.runtime, other.state) {}
    SmallCallback(SmallCallback&& other) noexcept
        : runtime(other.runtime), state(other.state) {
      other.state = nullptr;
    }
    ~SmallCallback() {
      if (!state) {
        return;
      }
      (void)runtime->info();
      ++state->destructions;
      --state->live_owners;
    }

    void operator()(
        const batching::GenerationUpdate& update,
        const RequestHandle&) const {
      ++state->calls;
      if (update.finish_reason) {
        ++state->terminals;
      }
    }
  };
  static_assert(sizeof(SmallCallback) == 2 * sizeof(void*));
  static_assert(std::is_nothrow_copy_constructible<SmallCallback>::value);
  auto input = request("session", std::make_shared<Events>(), 1);
  input.on_update = SmallCallback{runtime.get(), &small_callback_state};
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  input = {};
  ASSERT_TRUE(executor.executing.wait_for());
  EXPECT_EQ(small_callback_state.live_owners.load(), 1);
  const auto before_rejection = small_callback_state.destructions.load();
  auto rejected_input = request("rejected", std::make_shared<Events>(), 1);
  rejected_input.on_update =
      SmallCallback{runtime.get(), &small_callback_state};
  auto rejected = GenerationBridge::submit(*runtime, std::move(rejected_input));
  rejected_input = {};
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  EXPECT_GT(small_callback_state.destructions.load(), before_rejection);
  EXPECT_EQ(small_callback_state.live_owners.load(), 1);
  EXPECT_EQ(small_callback_state.calls.load(), 0);
  EXPECT_EQ(small_callback_state.terminals.load(), 0);
  const auto before_cleanup = small_callback_state.destructions.load();
  executor.executing.release();
  completed(handle);
  EXPECT_GT(small_callback_state.destructions.load(), before_cleanup);
  EXPECT_EQ(small_callback_state.live_owners.load(), 0);
  EXPECT_EQ(small_callback_state.calls.load(), 2);
  EXPECT_EQ(small_callback_state.terminals.load(), 1);
  EXPECT_FALSE(handle.error());
}

TEST_F(GenerationBridgeTest, OpaqueSuffixPreservesPendingTokenAndRawFeedback) {
  PreparedExecutor prepared_executor;
  ServingRuntime prepared_runtime(
      prepared_executor,
      batching::DecodeFirstScheduler::create(32, 4, 2),
      ServingRuntimeConfig{1, 128, 16});
  auto first_events = std::make_shared<Events>();
  auto first = accepted(GenerationBridge::submit(
      prepared_runtime, request("session", first_events, 1)));
  completed(first);
  ASSERT_TRUE(first.done());
  ASSERT_FALSE(first.error());
  ASSERT_TRUE(first_events->completion);
  ASSERT_TRUE(first_events->completion->position);
  ASSERT_EQ(first_events->updates.size(), 2u);
  ASSERT_EQ(first_events->updates.front().tokens.size(), 1u);
  const auto prior_position = *first_events->completion->position;
  EXPECT_EQ(prior_position, 2);
  const auto pending = first_events->updates.front().tokens.front();

  auto opaque = std::make_shared<TestPreparedInput>(3);
  auto next_events = std::make_shared<Events>();
  auto input = request("session", next_events, 2);
  input.delta = opaque;
  auto next =
      accepted(GenerationBridge::submit(prepared_runtime, std::move(input)));
  completed(next);
  prepared_runtime.shutdown();
  ASSERT_FALSE(next.error());
  ASSERT_TRUE(next_events->completion);
  ASSERT_EQ(next_events->updates.size(), 3u);
  EXPECT_EQ(next_events->terminals, 1);
  EXPECT_EQ(
      next_events->updates.back().finish_reason,
      batching::FinishReason::NewTokenLimit);
  EXPECT_TRUE(next_events->completion->current_session);
  EXPECT_EQ(
      next_events->completion->incarnation,
      first_events->completion->incarnation);
  EXPECT_EQ(next_events->completion->position, prior_position + 5);
  EXPECT_EQ(next_events->completion->metrics.n_prompt_tokens, 3);
  EXPECT_EQ(next_events->completion->metrics.n_prefilled_tokens, 4);
  EXPECT_EQ(next_events->completion->metrics.n_generated_tokens, 2);
  EXPECT_EQ(prepared_executor.opened().size(), 1u);
  EXPECT_EQ(prepared_executor.clones.load(), 0);

  const auto& slices = prepared_executor.slices;
  ASSERT_EQ(slices.size(), 5u);
  for (const auto& slice : slices) {
    EXPECT_EQ(slice.sid, slices.front().sid);
  }
  const auto& carried = slices[1];
  EXPECT_EQ(carried.position, prior_position);
  EXPECT_EQ(carried.offset, 0u);
  EXPECT_EQ(carried.size, 1u);
  EXPECT_FALSE(carried.produce_output);
  ASSERT_TRUE(std::holds_alternative<batching::TokenInputPtr>(carried.payload));
  EXPECT_EQ(
      *std::get<batching::TokenInputPtr>(carried.payload),
      (std::vector<batching::Token>{pending}));
  for (std::size_t i = 0; i < 2; ++i) {
    const auto& chunk = slices[2 + i];
    ASSERT_TRUE(
        std::holds_alternative<batching::PreparedInputPtr>(chunk.payload));
    EXPECT_EQ(std::get<batching::PreparedInputPtr>(chunk.payload), opaque);
    EXPECT_EQ(chunk.position, prior_position + 1);
    EXPECT_EQ(chunk.offset, i * 2);
    EXPECT_EQ(chunk.size, i == 0 ? 2u : 1u);
    EXPECT_EQ(chunk.produce_output, i == 1);
  }
  const auto& feedback = slices.back();
  EXPECT_EQ(feedback.position, prior_position + 4);
  EXPECT_EQ(feedback.offset, 0u);
  EXPECT_EQ(feedback.size, 1u);
  EXPECT_TRUE(feedback.produce_output);
  ASSERT_EQ(next_events->updates.front().tokens.size(), 1u);
  ASSERT_TRUE(
      std::holds_alternative<batching::TokenInputPtr>(feedback.payload));
  EXPECT_EQ(
      *std::get<batching::TokenInputPtr>(feedback.payload),
      next_events->updates.front().tokens);
}

TEST_F(GenerationBridgeTest, EmptyOpaqueSuffixRejectsBeforeAdmission) {
  ServingRuntimeConfig config{1, 128, 16};
  config.max_requests = 1;
  start(config);
  for (const auto& opaque :
       {batching::PreparedInputPtr{},
        batching::PreparedInputPtr{std::make_shared<TestPreparedInput>(0)}}) {
    auto events = std::make_shared<Events>();
    auto input = request("session", events);
    input.delta = opaque;
    auto result = GenerationBridge::submit(*runtime, std::move(input));
    ASSERT_TRUE(std::holds_alternative<ServingError>(result));
    EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
    EXPECT_TRUE(events->updates.empty());
    EXPECT_FALSE(events->prepared);
    EXPECT_FALSE(events->completion);
    EXPECT_TRUE(executor.opened().empty());
    EXPECT_EQ(executor.calls.load(), 0);
  }
  auto events = std::make_shared<Events>();
  auto handle = accepted(
      GenerationBridge::submit(*runtime, request("session", events, 1)));
  completed(handle);
  EXPECT_FALSE(handle.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
}

TEST_F(GenerationBridgeTest, DistinctSessionsReachOnePhysicalBatch) {
  start();
  opened("barrier");
  opened("first");
  opened("second");
  executor.executing.hold();
  auto barrier_events = std::make_shared<Events>();
  auto barrier = accepted(GenerationBridge::submit(
      *runtime, request("barrier", barrier_events, 1)));
  ASSERT_TRUE(executor.executing.wait_for());
  auto first_events = std::make_shared<Events>();
  auto second_events = std::make_shared<Events>();
  auto first = accepted(
      GenerationBridge::submit(*runtime, request("first", first_events)));
  auto second = accepted(
      GenerationBridge::submit(*runtime, request("second", second_events)));
  opened("barrier"); // FIFO control barrier: both generation starts were sent.
  executor.executing.release();
  completed(barrier);
  completed(first);
  completed(second);
  EXPECT_TRUE(executor.shared_batch.load());
  EXPECT_NE(first.id(), second.id());
  expect_terminal(first_events, batching::FinishReason::NewTokenLimit);
  expect_terminal(second_events, batching::FinishReason::NewTokenLimit);
}

TEST_F(
    GenerationBridgeTest,
    IdleEngineGenerationAllowsBusyDeliveryAndIsolatedCancellation) {
  start();
  opened("first");
  opened("second");
  executor.executing.hold();
  auto first_events = std::make_shared<Events>();
  auto first = accepted(
      GenerationBridge::submit(*runtime, request("first", first_events, 10)));
  ASSERT_TRUE(executor.executing.wait_for());
  auto busy_events = std::make_shared<Events>();
  auto busy = accepted(
      GenerationBridge::submit(*runtime, request("first", busy_events)));
  auto second_events = std::make_shared<Events>();
  auto second = accepted(
      GenerationBridge::submit(*runtime, request("second", second_events, 1)));
  completed(busy);
  EXPECT_FALSE(first.done());
  EXPECT_EQ(executor.calls.load(), 1);
  {
    std::lock_guard<std::mutex> lock(first_events->mutex);
    EXPECT_TRUE(first_events->updates.empty());
    EXPECT_FALSE(first_events->prepared);
  }
  ASSERT_TRUE(busy.error());
  EXPECT_EQ(busy.error()->code, ErrorCode::SessionBusy);
  first.cancel();
  executor.executing.release();
  completed(first);
  completed(second);
  EXPECT_FALSE(first.error());
  EXPECT_FALSE(second.error());
  expect_terminal(first_events, batching::FinishReason::Cancelled);
  expect_terminal(second_events, batching::FinishReason::NewTokenLimit);
  expect_terminal(busy_events, batching::FinishReason::Failed);
}

TEST_F(GenerationBridgeTest, SinkCancelsWithoutCallerHandleBinding) {
  executor.subsequent.hold();
  start();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 100);
  input.on_update = [this, events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    if (!update.finish_reason) {
      EXPECT_NE(handle.id(), 0u);
      handle.cancel();
      executor.subsequent.release();
    }
    events->record(update);
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  completed(handle);
  expect_terminal(events, batching::FinishReason::Cancelled);
  EXPECT_FALSE(handle.error());
  EXPECT_LE(executor.calls.load(), 2);
}

TEST_F(
    GenerationBridgeTest,
    InitializationFailureTerminatesAdmittedGeneration) {
  executor.initializing.hold();
  executor.fail_initialize = true;
  start();
  auto events = std::make_shared<Events>();
  auto handle =
      accepted(GenerationBridge::submit(*runtime, request("session", events)));
  executor.initializing.release();
  completed(handle);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::NotReady);
  expect_terminal(events, batching::FinishReason::Failed);
  std::lock_guard<std::mutex> lock(events->mutex);
  ASSERT_TRUE(events->completion->error);
  EXPECT_EQ(events->completion->error->code, ErrorCode::NotReady);
  EXPECT_FALSE(events->completion->current_session);
  EXPECT_TRUE(executor.opened().empty());
}

TEST_F(GenerationBridgeTest, QueuedCancellationDoesNotOpenSession) {
  executor.initializing.hold();
  start();
  auto barrier = runtime->open_session_async("barrier");
  ASSERT_TRUE(executor.initializing.wait_for());
  auto events = std::make_shared<Events>();
  auto handle = accepted(
      GenerationBridge::submit(*runtime, request("cancelled", events)));
  handle.cancel();
  executor.initializing.release();
  ASSERT_EQ(barrier.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(barrier.get());
  completed(handle);
  EXPECT_EQ(executor.opened().size(), 1u);
  EXPECT_EQ(executor.calls.load(), 0);
  expect_terminal(events, batching::FinishReason::Cancelled);
}

TEST_F(GenerationBridgeTest, CancellationBeforeBindingReleasesOpeningOwner) {
  executor.opening.hold();
  start();
  auto events = std::make_shared<Events>();
  auto handle = accepted(
      GenerationBridge::submit(*runtime, request(std::nullopt, events)));
  ASSERT_TRUE(executor.opening.wait_for());
  handle.cancel();
  executor.opening.release();
  completed(handle);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  EXPECT_EQ(executor.calls.load(), 0);
  expect_terminal(events, batching::FinishReason::Cancelled);
  runtime->shutdown();
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(GenerationBridgeTest, AnonymousSessionsAreIndependentAndEphemeral) {
  start();
  auto first_events = std::make_shared<Events>();
  auto second_events = std::make_shared<Events>();
  auto first = accepted(GenerationBridge::submit(
      *runtime, request(std::nullopt, first_events, 1)));
  auto second = accepted(GenerationBridge::submit(
      *runtime, request(std::nullopt, second_events, 1)));
  completed(first);
  completed(second);
  EXPECT_NE(first.id(), second.id());
  EXPECT_EQ(executor.opened().size(), 2u);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  runtime->shutdown();
  auto opens = executor.opened();
  auto closes = executor.closed();
  std::sort(opens.begin(), opens.end());
  std::sort(closes.begin(), closes.end());
  EXPECT_EQ(opens, closes);
}

TEST_F(GenerationBridgeTest, SpeculativeStopAndCommitPrecedeTerminalDelivery) {
  executor.tokens_per_prefill = 4;
  executor.stop_token = 999;
  executor.emit_before_stop = 1;
  start();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 10);
  input.config.stop_tokens = {999};
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  completed(handle);
  expect_terminal(events, batching::FinishReason::StopToken);
  std::lock_guard<std::mutex> lock(events->mutex);
  ASSERT_TRUE(events->completion);
  EXPECT_TRUE(events->completion->current_session);
  EXPECT_EQ(events->completion->request_id, handle.id());
  EXPECT_NE(events->completion->incarnation, 0u);
  EXPECT_EQ(events->completion->position, 4);
  EXPECT_EQ(events->completion->metrics.n_generated_tokens, 2);
  ASSERT_EQ(events->updates.size(), 2u);
  ASSERT_EQ(events->updates.front().tokens.size(), 2u);
  EXPECT_EQ(events->updates.front().tokens.back(), 999u);
  EXPECT_TRUE(events->updates.back().tokens.empty());
  EXPECT_EQ(events->tokens_seen_at_prepare, 2u);
}

TEST_F(
    GenerationBridgeTest,
    ReadyRequestsShareDeliveryFairlyAndReleaseCallbackCaptures) {
  constexpr std::size_t kRequests = 6;
  constexpr int kTokens = 3;
  ServingRuntimeConfig config{kRequests + 2, 128, 16};
  config.max_requests = kRequests + 1;
  start(config);
  opened("barrier");
  for (std::size_t i = 0; i < kRequests; ++i) {
    opened("ready-" + std::to_string(i));
  }
  const auto sessions = executor.opened();

  // Pause delivery only to accumulate ready mailboxes, not to require isolation
  // from a blocked callback. No other delivery is awaited until this gate
  // opens.
  sink_gate.hold();
  auto barrier_events = std::make_shared<Events>();
  auto barrier_input = request("barrier", barrier_events, 1);
  barrier_input.on_update = [this, barrier_events](
                                const batching::GenerationUpdate& update,
                                const RequestHandle&) {
    sink_gate.arrive();
    barrier_events->record(update);
  };
  auto barrier =
      accepted(GenerationBridge::submit(*runtime, std::move(barrier_input)));
  ASSERT_TRUE(sink_gate.wait_for());

  struct DeliveryOrder {
    std::mutex mutex;
    std::vector<std::size_t> requests;
  };
  auto order = std::make_shared<DeliveryOrder>();
  std::vector<RequestHandle> handles;
  std::vector<std::shared_ptr<Events>> events;
  std::vector<std::weak_ptr<Events>> captures;
  for (std::size_t i = 0; i < kRequests; ++i) {
    events.push_back(std::make_shared<Events>());
    captures.push_back(events.back());
    auto input = request("ready-" + std::to_string(i), events.back(), kTokens);
    input.on_update = [order, i, event = events.back()](
                          const batching::GenerationUpdate& update,
                          const RequestHandle&) {
      event->record(update);
      std::lock_guard<std::mutex> lock(order->mutex);
      order->requests.push_back(i);
    };
    handles.push_back(
        accepted(GenerationBridge::submit(*runtime, std::move(input))));
  }
  ASSERT_TRUE(wait_until(
      [&] { return executor.seen().size() == 1 + kRequests * kTokens; }));
  // A new physical open runs on the engine after its last observed execute,
  // so all token and settlement notifications are published before release.
  opened("engine-barrier");
  sink_gate.release();
  completed(barrier);
  for (std::size_t i = 0; i < kRequests; ++i) {
    completed(handles[i]);
    EXPECT_FALSE(handles[i].error());
    expect_terminal(events[i], batching::FinishReason::NewTokenLimit);
    std::lock_guard<std::mutex> lock(events[i]->mutex);
    EXPECT_EQ(events[i]->sink_threads.size(), 1u);
    EXPECT_EQ(events[i]->sink_thread, barrier_events->sink_thread);
    EXPECT_EQ(events[i]->tokens_seen_at_prepare, kTokens);
    ASSERT_TRUE(events[i]->completion);
    EXPECT_EQ(events[i]->completion->metrics.n_generated_tokens, kTokens);
    ASSERT_EQ(events[i]->updates.size(), kTokens + 1);
    std::mt19937_64 random(42);
    for (int token = 0; token < kTokens; ++token) {
      EXPECT_FALSE(events[i]->updates[token].finish_reason);
      EXPECT_EQ(
          events[i]->updates[token].tokens,
          (std::vector<batching::Token>{
              sessions[i + 1] * 1000 + random() % 1000}));
    }
    EXPECT_TRUE(events[i]->updates.back().tokens.empty());
  }
  {
    std::lock_guard<std::mutex> lock(order->mutex);
    ASSERT_EQ(order->requests.size(), kRequests * (kTokens + 1));
    // Each ready request gets one token turn before any gets a second turn.
    // Terminal acknowledgements may interleave differently on control.
    for (int round = 0; round < kTokens; ++round) {
      std::set<std::size_t> seen;
      for (std::size_t i = 0; i < kRequests; ++i) {
        seen.insert(order->requests[round * kRequests + i]);
      }
      EXPECT_EQ(seen.size(), kRequests);
    }
  }
  events.clear();
  for (const auto& capture : captures) {
    EXPECT_TRUE(capture.expired());
  }
  std::weak_ptr<DeliveryOrder> order_capture = order;
  order.reset();
  EXPECT_TRUE(order_capture.expired());
  // Retained handles must not retain callbacks or consume request admission.
  auto next = accepted(GenerationBridge::submit(
      *runtime, request("ready-0", std::make_shared<Events>(), 1)));
  completed(next);
  EXPECT_FALSE(next.error());
}

TEST_F(
    GenerationBridgeTest,
    SaturatedAdmissionRejectsWithoutCallbackAndReleasesCompletedSlots) {
  ServingRuntimeConfig config{3, 128, 16};
  config.max_requests = 2;
  start(config);
  opened("first");
  opened("second");
  executor.executing.hold();
  executor.subsequent.hold();
  sink_gate.hold();
  auto first_input = request("first", std::make_shared<Events>(), 1);
  first_input.on_update = [this](
                              const batching::GenerationUpdate&,
                              const RequestHandle&) { sink_gate.arrive(); };
  auto second_input = request("second", std::make_shared<Events>(), 1);
  auto first =
      accepted(GenerationBridge::submit(*runtime, std::move(first_input)));
  auto second =
      accepted(GenerationBridge::submit(*runtime, std::move(second_input)));
  ASSERT_TRUE(executor.executing.wait_for());
  auto rejected_events = std::make_shared<Events>();
  auto rejected = GenerationBridge::submit(
      *runtime, request("rejected", rejected_events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  EXPECT_TRUE(rejected_events->updates.empty());
  EXPECT_EQ(executor.opened().size(), 2u);
  executor.executing.release();
  ASSERT_TRUE(sink_gate.wait_for());
  EXPECT_FALSE(first.done());
  auto still_rejected = GenerationBridge::submit(
      *runtime, request("rejected", rejected_events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(still_rejected));
  EXPECT_EQ(
      std::get<ServingError>(still_rejected).code, ErrorCode::CapacityExceeded);
  sink_gate.release();
  completed(first);
  executor.subsequent.release();
  completed(second);
  auto next = accepted(GenerationBridge::submit(
      *runtime, request("rejected", rejected_events, 1)));
  completed(next);
  EXPECT_FALSE(next.error());
}

TEST_F(
    GenerationBridgeTest,
    TerminalCallbackCanSubmitSameSessionBeforeReturning) {
  watch_callbacks();
  ServingRuntimeConfig config{1, 128, 16};
  config.max_requests = 1;
  start(config);
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto next_events = std::make_shared<Events>();
  auto submitted = std::make_shared<std::promise<SubmissionResult>>();
  auto followup = submitted->get_future();
  auto input = request("session", events, 1);
  input.on_update = [this, events, next_events, submitted](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    events->record(update);
    if (update.finish_reason) {
      EXPECT_FALSE(handle.done());
      submitted->set_value(GenerationBridge::submit(
          *runtime, request("session", next_events, 1)));
      sink_gate.arrive();
      EXPECT_FALSE(handle.done());
    }
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_EQ(followup.wait_for(kTimeout), std::future_status::ready);
  auto result = followup.get();
  ASSERT_TRUE(std::holds_alternative<RequestHandle>(result));
  auto next = std::get<RequestHandle>(std::move(result));
  EXPECT_NE(next.id(), handle.id());
  // Control and engine can bind and execute, but the sole dispatcher is held.
  ASSERT_TRUE(wait_until([&] { return executor.seen().size() == 2; }));
  EXPECT_FALSE(handle.done());
  EXPECT_FALSE(next.done());
  EXPECT_FALSE(next.error());
  const auto seen = executor.seen();
  EXPECT_EQ(seen[0].session, seen[1].session);
  EXPECT_EQ(executor.opened().size(), 1u);
  {
    std::lock_guard<std::mutex> lock(events->mutex);
    ASSERT_TRUE(events->completion);
    ASSERT_TRUE(events->completion->position);
    EXPECT_EQ(seen[1].position, *events->completion->position);
    EXPECT_GT(seen[1].position, seen[0].position);
  }
  {
    std::lock_guard<std::mutex> lock(next_events->mutex);
    EXPECT_TRUE(next_events->updates.empty());
  }
  sink_gate.release();
  completed(handle);
  completed(next);
  EXPECT_FALSE(handle.error());
  EXPECT_FALSE(next.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
  expect_terminal(next_events, batching::FinishReason::NewTokenLimit);
}

TEST_F(
    GenerationBridgeTest,
    ControlFinalizedRequestRetainsAdmissionUntilTerminalSelection) {
  watch_callbacks();
  ServingRuntimeConfig config{2, 128, 16};
  config.max_requests = 2;
  start(config);
  opened("target");
  opened("blocker");
  executor.executing.hold();
  executor.subsequent.hold();
  other_sink_gate.hold();
  sink_gate.hold();
  auto target_events = std::make_shared<Events>();
  auto target_input = request("target", target_events, 1);
  target_input.on_complete = [this, commit = target_input.on_complete](
                                 const GenerationCompletion& completion) {
    commit(completion);
    other_sink_gate.arrive();
  };
  auto target =
      accepted(GenerationBridge::submit(*runtime, std::move(target_input)));
  ASSERT_TRUE(executor.executing.wait_for());
  auto blocker_events = std::make_shared<Events>();
  auto blocker_input = request("blocker", blocker_events, 1);
  blocker_input.on_update = [this, blocker_events](
                                const batching::GenerationUpdate& update,
                                const RequestHandle&) {
    blocker_events->record(update);
    if (!update.finish_reason) {
      sink_gate.arrive();
    }
  };
  auto blocker =
      accepted(GenerationBridge::submit(*runtime, std::move(blocker_input)));
  opened("blocker"); // Both starts precede the gated target finalizer.
  executor.executing.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  ASSERT_TRUE(executor.subsequent.wait_for());
  executor.subsequent.release();
  ASSERT_TRUE(sink_gate.wait_for());
  other_sink_gate.release();
  opened("target"); // Control finalized target while delivery remains blocked.
  EXPECT_FALSE(target.done());
  EXPECT_FALSE(blocker.done());
  {
    std::lock_guard<std::mutex> lock(target_events->mutex);
    ASSERT_TRUE(target_events->completion);
    EXPECT_TRUE(target_events->completion->current_session);
    EXPECT_EQ(target_events->terminals, 0);
  }
  // Finalization cannot admit an unbounded backlog of undelivered terminals.
  auto rejected_events = std::make_shared<Events>();
  for (int attempt = 0; attempt < 3; ++attempt) {
    auto rejected = GenerationBridge::submit(
        *runtime, request("target", rejected_events, 1));
    ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
    EXPECT_EQ(
        std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  }
  {
    std::lock_guard<std::mutex> lock(rejected_events->mutex);
    EXPECT_TRUE(rejected_events->updates.empty());
    EXPECT_FALSE(rejected_events->completion);
  }
  sink_gate.release();
  completed(target);
  completed(blocker);
  expect_terminal(target_events, batching::FinishReason::NewTokenLimit);
  expect_terminal(blocker_events, batching::FinishReason::NewTokenLimit);
}

TEST_F(GenerationBridgeTest, FullControlQueueDoesNotLeakRequestAdmission) {
  executor.opening.hold();
  ServingRuntimeConfig config{3, 128, 1};
  config.max_requests = 1;
  start(config);
  auto barrier = runtime->open_session_async("barrier");
  ASSERT_TRUE(executor.opening.wait_for());
  auto events = std::make_shared<Events>();
  auto rejected =
      GenerationBridge::submit(*runtime, request("request", events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  executor.opening.release();
  ASSERT_EQ(barrier.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(barrier.get());
  auto next = accepted(
      GenerationBridge::submit(*runtime, request("request", events, 1)));
  completed(next);
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
}

TEST_F(
    GenerationBridgeTest,
    OverflowIsBoundedAndReservedTerminalReportsFailure) {
  ServingRuntimeConfig config{2, 128, 16};
  config.max_events_per_request = 1;
  config.max_tokens_per_request = 2;
  executor.subsequent.hold();
  start(config);
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 100);
  input.on_update = [this, events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle&) {
    if (!update.finish_reason) {
      sink_gate.arrive();
    }
    events->record(update);
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_TRUE(executor.subsequent.wait_for());
  executor.subsequent.release();
  ASSERT_TRUE(wait_until([&] { return handle.error().has_value(); }));
  EXPECT_EQ(handle.error()->code, ErrorCode::CapacityExceeded);
  sink_gate.release();
  completed(handle);
  expect_terminal(events, batching::FinishReason::Failed);
  EXPECT_LT(executor.calls.load(), 100);
}

TEST_F(
    GenerationBridgeTest,
    OversizedSpeculativeTerminalCannotBypassTokenLimit) {
  ServingRuntimeConfig config{2, 128, 16};
  config.max_tokens_per_request = 2;
  executor.tokens_per_prefill = 4;
  start(config);
  auto events = std::make_shared<Events>();
  auto handle = accepted(
      GenerationBridge::submit(*runtime, request("session", events, 4)));
  completed(handle);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::CapacityExceeded);
  expect_terminal(events, batching::FinishReason::Failed);
  std::lock_guard<std::mutex> lock(events->mutex);
  EXPECT_TRUE(events->updates.back().tokens.empty());
}

TEST_P(GenerationFenceTest, AckWaitsForTokensTerminalAndCaptureCleanup) {
  watch_callbacks();
  start();
  opened("session");
  opened("other");
  const auto old_session = executor.opened().front();
  const auto other_session = executor.opened().back();
  auto acked = std::make_shared<std::atomic<bool>>(false);
  executor.on_execute = [acked, old_session, other_session](const auto& batch) {
    for (const auto& item : batch.inputs) {
      if (item.sid != old_session && item.sid != other_session) {
        EXPECT_TRUE(acked->load());
      }
    }
  };
  executor.subsequent.hold();
  sink_gate.hold();
  other_sink_gate.hold();
  cleanup_gate.hold();
  auto cleaned = std::make_shared<std::atomic<bool>>(false);
  auto capture = std::shared_ptr<int>(new int(0), [this, cleaned](int* value) {
    (void)runtime->info();
    EXPECT_FALSE(cleanup_handle.done());
    cleanup_gate.arrive();
    cleaned->store(true);
    delete value;
  });
  auto old_events = std::make_shared<Events>();
  auto input = request("session", old_events, 2);
  input.on_update = [this, old_events, capture](
                        const batching::GenerationUpdate& update,
                        const RequestHandle&) {
    (void)capture;
    old_events->record(update);
    if (update.finish_reason) {
      other_sink_gate.arrive();
    } else {
      sink_gate.arrive();
    }
  };
  cleanup_handle =
      accepted(GenerationBridge::submit(*runtime, std::move(input)));
  input = {};
  capture.reset();
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_TRUE(executor.subsequent.wait_for());

  auto promise = std::make_shared<std::promise<serving::LifecycleResult>>();
  auto ack = promise->get_future();
  fence("session", [this, cleaned, acked, promise](auto error) {
    EXPECT_TRUE(cleanup_handle.done());
    EXPECT_TRUE(cleaned->load());
    (void)runtime->info();
    acked->store(true);
    promise->set_value(std::move(error));
  });
  auto replacement_events = std::make_shared<Events>();
  auto replacement_input = request("session", replacement_events, 1);
  replacement_input.on_prepare_complete =
      [acked, prepare = replacement_input.on_prepare_complete](
          const GenerationCompletion& completion) {
        EXPECT_TRUE(acked->load());
        prepare(completion);
      };
  auto replacement = accepted(
      GenerationBridge::submit(*runtime, std::move(replacement_input)));
  executor.subsequent.release();
  opened("other"); // Control must skip the fenced key, including its followup.
  auto other_events = std::make_shared<Events>();
  auto other = accepted(
      GenerationBridge::submit(*runtime, request("other", other_events, 1)));
  ASSERT_TRUE(wait_until([&] {
    const auto seen = executor.seen();
    return std::any_of(seen.begin(), seen.end(), [&](const auto& item) {
      return item.session == other_session;
    });
  }));
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_FALSE(cleanup_handle.done());
  EXPECT_FALSE(replacement.done());
  EXPECT_EQ(runtime->info().active_sessions, GetParam() ? 2u : 1u);
  EXPECT_EQ(executor.opened().size(), GetParam() ? 3u : 2u);
  {
    // Delivery is shared: unrelated execution progresses, not its blocked sink.
    std::lock_guard<std::mutex> lock(other_events->mutex);
    EXPECT_TRUE(other_events->updates.empty());
  }

  sink_gate.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  opened("other"); // Finalizers must also run while the fence is pending.
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_FALSE(cleanup_handle.done());
  {
    std::lock_guard<std::mutex> lock(old_events->mutex);
    ASSERT_TRUE(old_events->completion);
    EXPECT_FALSE(old_events->completion->current_session);
    EXPECT_FALSE(old_events->completion->position);
    EXPECT_EQ(old_events->terminals, 1);
  }
  other_sink_gate.release();
  ASSERT_TRUE(cleanup_gate.wait_for());
  opened("other");
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_FALSE(cleanup_handle.done());
  EXPECT_FALSE(cleaned->load());
  EXPECT_EQ(executor.opened().size(), GetParam() ? 3u : 2u);
  cleanup_gate.release();
  ASSERT_EQ(ack.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  EXPECT_TRUE(cleaned->load());
  completed(cleanup_handle);
  completed(replacement);
  completed(other);
  EXPECT_FALSE(replacement.error());
  EXPECT_FALSE(other.error());
  expect_terminal(replacement_events, batching::FinishReason::NewTokenLimit);
  expect_terminal(other_events, batching::FinishReason::NewTokenLimit);
  EXPECT_EQ(executor.opened().size(), 3u); // Apply each transition only once.
}

TEST_P(GenerationFenceTest, RetiringTerminalCanEnqueueFenceAndFollowup) {
  watch_callbacks();
  ServingRuntimeConfig config{2, 128, 16};
  config.max_requests = 1;
  start(config);
  opened("other");
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto next_events = std::make_shared<Events>();
  auto acked = std::make_shared<std::atomic<bool>>(false);
  auto ack_promise = std::make_shared<std::promise<serving::LifecycleResult>>();
  auto ack = ack_promise->get_future();
  auto submitted = std::make_shared<std::promise<SubmissionResult>>();
  auto followup = submitted->get_future();
  auto input = request("session", events, 1);
  input.on_update = [this, events, next_events, acked, ack_promise, submitted](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    events->record(update);
    if (update.finish_reason) {
      EXPECT_FALSE(handle.done());
      (void)runtime->info();
      // Neither the admission registry nor the active claim contains this
      // request now. Submission from its terminal must remain nonblocking.
      fence("session", [this, handle, acked, ack_promise](auto error) {
        EXPECT_TRUE(handle.done());
        (void)runtime->info();
        acked->store(true);
        ack_promise->set_value(std::move(error));
      });
      auto next = request("session", next_events, 1);
      next.on_update = [next_events, acked](
                           const batching::GenerationUpdate& next_update,
                           const RequestHandle&) {
        EXPECT_TRUE(acked->load());
        next_events->record(next_update);
      };
      submitted->set_value(GenerationBridge::submit(*runtime, std::move(next)));
      sink_gate.arrive();
    }
  };
  auto old = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_EQ(followup.wait_for(kTimeout), std::future_status::ready);
  auto next = accepted(followup.get());
  ASSERT_NE(next.id(), 0u);
  opened("other");
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(executor.seen().size(), 1u);
  EXPECT_FALSE(old.done());
  EXPECT_FALSE(next.done());
  sink_gate.release();
  ASSERT_EQ(ack.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  completed(old);
  completed(next);
  EXPECT_FALSE(next.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
  expect_terminal(next_events, batching::FinishReason::NewTokenLimit);
  const auto seen = executor.seen();
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_NE(seen[0].session, seen[1].session);
  EXPECT_EQ(seen[1].position, 0);
}

TEST_P(
    GenerationFenceTest,
    QueuedBusyPredecessorsAreFencedButLaterRequestIsNot) {
  watch_callbacks();
  start();
  opened("session");
  opened("other");
  sink_gate.hold();
  other_sink_gate.hold();
  cleanup_gate.hold();
  auto old_input = request("session", std::make_shared<Events>(), 1);
  old_input.on_update =
      [this](const batching::GenerationUpdate& update, const RequestHandle&) {
        if (!update.finish_reason) {
          sink_gate.arrive();
        }
      };
  auto old = accepted(GenerationBridge::submit(*runtime, std::move(old_input)));
  ASSERT_TRUE(sink_gate.wait_for());
  auto predecessors = std::make_shared<std::vector<RequestHandle>>();
  predecessors->push_back(old);
  for (int i = 0; i < 2; ++i) {
    auto busy = request("session", std::make_shared<Events>(), 1);
    busy.on_update = [this](
                         const batching::GenerationUpdate& update,
                         const RequestHandle& handle) {
      EXPECT_EQ(update.finish_reason, batching::FinishReason::Failed);
      ASSERT_TRUE(handle.error());
      EXPECT_EQ(handle.error()->code, ErrorCode::SessionBusy);
      other_sink_gate.arrive();
    };
    predecessors->push_back(
        accepted(GenerationBridge::submit(*runtime, std::move(busy))));
  }
  opened("other"); // Both Busy results precede the fence but await delivery.
  for (std::size_t i = 1; i < predecessors->size(); ++i) {
    ASSERT_TRUE((*predecessors)[i].error());
    EXPECT_EQ((*predecessors)[i].error()->code, ErrorCode::SessionBusy);
  }
  auto promise = std::make_shared<std::promise<serving::LifecycleResult>>();
  auto ack = promise->get_future();
  fence("session", [predecessors, promise](auto error) {
    for (const auto& handle : *predecessors) {
      EXPECT_TRUE(handle.done());
    }
    promise->set_value(std::move(error));
  });
  auto later_input = request("session", std::make_shared<Events>(), 1);
  later_input.on_update =
      [this](const batching::GenerationUpdate& update, const RequestHandle&) {
        if (update.finish_reason) {
          cleanup_gate.arrive();
        }
      };
  auto later =
      accepted(GenerationBridge::submit(*runtime, std::move(later_input)));
  auto closed_promise =
      std::make_shared<std::promise<serving::LifecycleResult>>();
  auto closed = closed_promise->get_future();
  runtime->close_session_async("session", [later, closed_promise](auto error) {
    EXPECT_TRUE(later.done());
    closed_promise->set_value(std::move(error));
  });
  opened("other");
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(
      closed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  sink_gate.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  other_sink_gate.release();
  ASSERT_EQ(ack.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(ack.get());
  ASSERT_TRUE(cleanup_gate.wait_for());
  opened("other");
  EXPECT_FALSE(later.done());
  EXPECT_EQ(
      closed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  cleanup_gate.release();
  ASSERT_EQ(closed.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(closed.get());
  for (const auto& handle : *predecessors) {
    completed(handle);
  }
  completed(later);
  EXPECT_FALSE(later.error());
  EXPECT_EQ(runtime->info().active_sessions, 1u);
}

TEST_P(GenerationFenceTest, MissingKeyFenceRetainsPermitsThroughCleanup) {
  watch_callbacks();
  ServingRuntimeConfig config{1, 128, 2};
  config.max_requests = 1;
  start(config);
  opened("resident");
  sink_gate.hold();
  other_sink_gate.hold();
  auto capture = std::shared_ptr<int>(new int(0), [this](int* value) {
    (void)runtime->info();
    other_sink_gate.arrive();
    delete value;
  });
  auto input = request("missing", std::make_shared<Events>(), 1);
  input.on_update = [this, capture](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    (void)capture;
    EXPECT_EQ(update.finish_reason, batching::FinishReason::Failed);
    ASSERT_TRUE(handle.error());
    EXPECT_EQ(handle.error()->code, ErrorCode::CapacityExceeded);
    sink_gate.arrive();
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  input = {};
  capture.reset();
  ASSERT_TRUE(sink_gate.wait_for());
  auto ack = fence("missing");
  opened(
      "resident"); // The missing-key result is computed but not acknowledged.
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  auto later_close = runtime->close_session_async("missing");
  auto rejected = runtime->open_session_async("resident");
  ASSERT_EQ(rejected.wait_for(kTimeout), std::future_status::ready);
  auto error = rejected.get();
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, ErrorCode::CapacityExceeded);
  auto rejected_events = std::make_shared<Events>();
  auto rejected_request = GenerationBridge::submit(
      *runtime, request("resident", rejected_events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected_request));
  EXPECT_EQ(
      std::get<ServingError>(rejected_request).code,
      ErrorCode::CapacityExceeded);
  EXPECT_TRUE(rejected_events->updates.empty());
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  sink_gate.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  EXPECT_FALSE(handle.done());
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(
      later_close.wait_for(std::chrono::seconds(0)),
      std::future_status::timeout);
  auto still_rejected = runtime->open_session_async("resident");
  ASSERT_EQ(still_rejected.wait_for(kTimeout), std::future_status::ready);
  error = still_rejected.get();
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, ErrorCode::CapacityExceeded);
  other_sink_gate.release();
  ASSERT_EQ(ack.wait_for(kTimeout), std::future_status::ready);
  auto result = ack.get();
  if (GetParam()) {
    ASSERT_TRUE(result);
    EXPECT_EQ(result->code, ErrorCode::SessionNotFound);
  } else {
    EXPECT_FALSE(result);
  }
  ASSERT_EQ(later_close.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(later_close.get());
  completed(handle);
  opened("resident"); // Both pending-operation permits were returned.
  EXPECT_EQ(executor.opened().size(), 1u);
}

TEST_F(GenerationBridgeTest, ShutdownSettlesHandlesAndWaitsForReturningSinks) {
  start();
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_update = [this, events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle&) {
    sink_gate.arrive();
    events->record(update);
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  std::atomic<bool> stopped{false};
  std::thread stopping([&] {
    runtime->shutdown();
    stopped.store(true);
  });
  const bool stopping_admission =
      wait_until([&] { return !runtime->info().ready; });
  EXPECT_TRUE(stopping_admission);
  EXPECT_FALSE(stopped.load());
  auto rejected = GenerationBridge::submit(
      *runtime, request("late", std::make_shared<Events>()));
  EXPECT_TRUE(std::holds_alternative<ServingError>(rejected));
  sink_gate.release();
  stopping.join();
  completed(handle);
  runtime.reset();
  handle.cancel();
  EXPECT_TRUE(handle.done());
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(
    GenerationBridgeTest,
    ShutdownAndWaitIncludeSelectedTerminalAndUnlockedCaptureCleanup) {
  watch_callbacks();
  ServingRuntimeConfig config{1, 128, 16};
  config.max_requests = 1;
  start(config);
  sink_gate.hold();
  auto cleaned = std::make_shared<std::atomic<bool>>(false);
  auto capture = std::shared_ptr<int>(new int(0), [this, cleaned](int* value) {
    // Both runtime and request locks must be released during capture cleanup.
    (void)runtime->info();
    EXPECT_FALSE(cleanup_handle.done());
    other_sink_gate.arrive();
    EXPECT_FALSE(cleanup_handle.done());
    cleaned->store(true);
    delete value;
  });
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_update = [this, events, capture](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    (void)capture;
    events->record(update);
    if (update.finish_reason) {
      EXPECT_FALSE(handle.done());
      sink_gate.arrive();
      EXPECT_FALSE(handle.done());
      EXPECT_EQ(update.finish_reason, batching::FinishReason::NewTokenLimit);
    }
  };
  cleanup_handle =
      accepted(GenerationBridge::submit(*runtime, std::move(input)));
  input = {}; // Do not let moved-from callback captures delay destruction.
  ASSERT_TRUE(sink_gate.wait_for());
  other_sink_gate.hold();
  capture.reset();
  auto promise = std::make_shared<std::promise<serving::LifecycleResult>>();
  auto closed = promise->get_future();
  runtime->close_session_async("session", [this, cleaned, promise](auto error) {
    EXPECT_TRUE(cleanup_handle.done());
    EXPECT_TRUE(cleaned->load());
    promise->set_value(std::move(error));
  });
  auto reset = runtime->reset_session_async("session");
  auto barrier = runtime->close_session_async("other");
  ASSERT_EQ(barrier.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(barrier.get());
  EXPECT_EQ(
      closed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(
      reset.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  // This is the only request: terminal selection has emptied the registry.
  auto started = std::make_shared<std::atomic<int>>(0);
  auto stopped = std::make_shared<std::atomic<int>>(0);
  auto waited = std::make_shared<std::atomic<bool>>(false);
  for (int i = 0; i < 2; ++i) {
    callback_callers.emplace_back([this, started, stopped] {
      ++*started;
      runtime->shutdown();
      ++*stopped;
    });
  }
  callback_callers.emplace_back([handle = cleanup_handle, started, waited] {
    ++*started;
    handle.wait();
    waited->store(true);
  });
  ASSERT_TRUE(wait_until([&] { return started->load() == 3; }));
  ASSERT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  EXPECT_EQ(stopped->load(), 0);
  EXPECT_FALSE(waited->load());
  EXPECT_FALSE(cleanup_handle.done());
  EXPECT_FALSE(cleaned->load());
  sink_gate.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  EXPECT_EQ(stopped->load(), 0);
  EXPECT_FALSE(waited->load());
  EXPECT_FALSE(cleanup_handle.done());
  EXPECT_FALSE(cleaned->load());
  EXPECT_EQ(
      closed.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  EXPECT_EQ(
      reset.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  other_sink_gate.release();
  ASSERT_TRUE(
      wait_until([&] { return stopped->load() == 2 && waited->load(); }));
  for (auto* future : {&closed, &reset}) {
    ASSERT_EQ(future->wait_for(kTimeout), std::future_status::ready);
    auto error = future->get();
    ASSERT_TRUE(error);
    EXPECT_EQ(error->code, ErrorCode::NotReady);
  }
  EXPECT_TRUE(cleaned->load());
  completed(cleanup_handle);
  EXPECT_FALSE(cleanup_handle.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
  EXPECT_EQ(executor.closed(), executor.opened());
}

TEST_F(
    GenerationBridgeTest,
    ConcurrentShutdownSettlesActiveAndQueuedGenerations) {
  start();
  opened("first");
  opened("second");
  executor.executing.hold();
  auto first_events = std::make_shared<Events>();
  auto second_events = std::make_shared<Events>();
  auto third_events = std::make_shared<Events>();
  auto first = accepted(
      GenerationBridge::submit(*runtime, request("first", first_events, 100)));
  ASSERT_TRUE(executor.executing.wait_for());
  auto second = accepted(GenerationBridge::submit(
      *runtime, request("second", second_events, 100)));
  opened("second");
  auto third = accepted(GenerationBridge::submit(
      *runtime, request(std::nullopt, third_events, 100)));
  std::thread stopping([&] { runtime->shutdown(); });
  std::thread also_stopping([&] { runtime->shutdown(); });
  EXPECT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  executor.executing.release();
  stopping.join();
  also_stopping.join();
  completed(first);
  completed(second);
  completed(third);
  expect_terminal(first_events, batching::FinishReason::Cancelled);
  expect_terminal(second_events, batching::FinishReason::Cancelled);
  expect_terminal(third_events, batching::FinishReason::Cancelled);
  EXPECT_EQ(runtime->info().active_sessions, 0u);
}

#if ET_HAS_EXCEPTIONS
TEST_P(GenerationFenceTest, ThrowingSinkFenceDrainsCleanupDuringShutdown) {
  watch_callbacks();
  start();
  opened("session");
  opened("other");
  sink_gate.hold();
  other_sink_gate.hold();
  auto cleaned = std::make_shared<std::atomic<bool>>(false);
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto capture = std::shared_ptr<int>(new int(0), [this, cleaned](int* value) {
    (void)runtime->info();
    other_sink_gate.arrive();
    cleaned->store(true);
    delete value;
  });
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_update = [this, capture, calls](
                        const batching::GenerationUpdate& update,
                        const RequestHandle&) {
    (void)capture;
    EXPECT_FALSE(update.finish_reason);
    ++*calls;
    sink_gate.arrive();
    throw 1;
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  input = {};
  capture.reset();
  ASSERT_TRUE(sink_gate.wait_for());
  auto promise = std::make_shared<std::promise<serving::LifecycleResult>>();
  auto ack = promise->get_future();
  fence("session", [handle, cleaned, promise](auto error) {
    EXPECT_TRUE(handle.done());
    EXPECT_TRUE(cleaned->load());
    promise->set_value(std::move(error));
  });
  opened("other");
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  sink_gate.release();
  ASSERT_TRUE(other_sink_gate.wait_for());
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  EXPECT_FALSE(handle.done());
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  auto stopped = std::make_shared<std::atomic<bool>>(false);
  callback_callers.emplace_back([this, stopped] {
    runtime->shutdown();
    stopped->store(true);
  });
  ASSERT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  EXPECT_FALSE(stopped->load());
  EXPECT_EQ(ack.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
  other_sink_gate.release();
  ASSERT_EQ(ack.wait_for(kTimeout), std::future_status::ready);
  auto error = ack.get();
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, ErrorCode::NotReady);
  ASSERT_TRUE(wait_until([&] { return stopped->load(); }));
  completed(handle);
  EXPECT_TRUE(cleaned->load());
  EXPECT_EQ(calls->load(), 1); // A throwing sink suppresses terminal delivery.
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  std::lock_guard<std::mutex> lock(events->mutex);
  ASSERT_TRUE(events->completion);
  EXPECT_EQ(
      events->completion->terminal.finish_reason,
      batching::FinishReason::Failed);
  EXPECT_FALSE(events->completion->current_session);
}

TEST_F(
    GenerationBridgeTest,
    ThrowingTerminalPreservesSuccessAndResidentSessionPosition) {
  watch_callbacks();
  ServingRuntimeConfig config{2, 128, 16};
  config.max_requests = 1;
  start(config);
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_update = [this, events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    events->record(update);
    if (update.finish_reason) {
      EXPECT_EQ(update.finish_reason, batching::FinishReason::NewTokenLimit);
      EXPECT_FALSE(handle.error());
      sink_gate.arrive();
      EXPECT_FALSE(handle.done());
      EXPECT_FALSE(handle.error());
      EXPECT_EQ(update.finish_reason, batching::FinishReason::NewTokenLimit);
      throw 1;
    }
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  handle.cancel(); // The successful control outcome is already frozen.
  EXPECT_FALSE(handle.done());
  EXPECT_FALSE(handle.error());
  sink_gate.release();
  completed(handle);
  EXPECT_FALSE(handle.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_TRUE(executor.closed().empty());

  auto next_events = std::make_shared<Events>();
  auto next = accepted(
      GenerationBridge::submit(*runtime, request("session", next_events, 1)));
  completed(next);
  EXPECT_FALSE(next.error());
  expect_terminal(next_events, batching::FinishReason::NewTokenLimit);
  const auto seen = executor.seen();
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0].session, seen[1].session);
  EXPECT_EQ(executor.opened().size(), 1u);
  {
    std::lock_guard<std::mutex> lock(events->mutex);
    ASSERT_TRUE(events->completion);
    EXPECT_TRUE(events->completion->current_session);
    EXPECT_FALSE(events->completion->error);
    ASSERT_TRUE(events->completion->position);
    EXPECT_EQ(seen[1].position, *events->completion->position);
    EXPECT_GT(seen[1].position, seen[0].position);
  }
  auto unrelated_events = std::make_shared<Events>();
  auto unrelated = accepted(GenerationBridge::submit(
      *runtime, request("unrelated", unrelated_events, 1)));
  completed(unrelated);
  EXPECT_FALSE(unrelated.error());
  expect_terminal(unrelated_events, batching::FinishReason::NewTokenLimit);
  handle.cancel();
  runtime->shutdown();
  EXPECT_FALSE(handle.error());
  expect_terminal(events, batching::FinishReason::NewTokenLimit);
}

TEST_F(
    GenerationBridgeTest,
    ThrowingTerminalPreservesCapacityFailureAcrossCancelAndShutdown) {
  watch_callbacks();
  ServingRuntimeConfig config{1, 128, 16};
  config.max_requests = 1;
  start(config);
  opened("resident");
  sink_gate.hold();
  auto events = std::make_shared<Events>();
  auto input = request("no-slot", events, 1);
  input.on_update = [this, events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle& handle) {
    events->record(update);
    EXPECT_EQ(update.finish_reason, batching::FinishReason::Failed);
    ASSERT_TRUE(handle.error());
    EXPECT_EQ(handle.error()->code, ErrorCode::CapacityExceeded);
    sink_gate.arrive();
    throw 1;
  };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  const auto error = handle.error();
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, ErrorCode::CapacityExceeded);
  handle.cancel();
  auto stopped = std::make_shared<std::atomic<bool>>(false);
  callback_callers.emplace_back([this, stopped] {
    runtime->shutdown();
    stopped->store(true);
  });
  ASSERT_TRUE(wait_until([&] { return !runtime->info().ready; }));
  EXPECT_FALSE(handle.done());
  EXPECT_FALSE(stopped->load());
  sink_gate.release();
  completed(handle);
  ASSERT_TRUE(wait_until([&] { return stopped->load(); }));
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, error->code);
  EXPECT_EQ(handle.error()->message, error->message);
  expect_terminal(events, batching::FinishReason::Failed);
  std::lock_guard<std::mutex> lock(events->mutex);
  ASSERT_TRUE(events->completion);
  ASSERT_TRUE(events->completion->error);
  EXPECT_EQ(events->completion->error->code, error->code);
  EXPECT_EQ(events->completion->error->message, error->message);
  EXPECT_FALSE(events->completion->current_session);
  EXPECT_EQ(
      events->completion->terminal.finish_reason,
      batching::FinishReason::Failed);
}

TEST_F(
    GenerationBridgeTest,
    ThrowingPreparationStillCommitsFailureAndTerminates) {
  start();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_prepare_complete = [](const GenerationCompletion&) { throw 1; };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  completed(handle);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  std::lock_guard<std::mutex> lock(events->mutex);
  ASSERT_TRUE(events->completion);
  ASSERT_TRUE(events->completion->error);
  EXPECT_EQ(events->completion->error->code, ErrorCode::Internal);
  EXPECT_EQ(events->terminals, 1);
  EXPECT_TRUE(events->terminal_saw_commit);
  EXPECT_EQ(
      events->updates.back().finish_reason, batching::FinishReason::Failed);
}

TEST_F(
    GenerationBridgeTest,
    ThrowingCommitInvalidatesSessionAndStillDeliversTerminal) {
  start();
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 1);
  input.on_complete = [](const GenerationCompletion&) { throw 1; };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  completed(handle);
  ASSERT_TRUE(handle.error());
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  {
    std::lock_guard<std::mutex> lock(events->mutex);
    EXPECT_EQ(events->terminals, 1);
    EXPECT_EQ(
        events->updates.back().finish_reason, batching::FinishReason::Failed);
  }
  auto future = runtime->open_session_async("session");
  ASSERT_EQ(future.wait_for(kTimeout), std::future_status::ready);
  auto error = future.get();
  ASSERT_TRUE(error);
  EXPECT_EQ(error->code, ErrorCode::NotReady);
}

TEST_F(
    GenerationBridgeTest,
    ThrowingSinkRetainsOwnershipUntilSettlementAndAllowsLocalRejection) {
  ServingRuntimeConfig config{1, 128, 16};
  config.max_requests = 2;
  executor.subsequent.hold();
  start(config);
  sink_gate.hold();
  auto calls = std::make_shared<std::atomic<int>>(0);
  auto events = std::make_shared<Events>();
  auto input = request("session", events, 10);
  input.on_update =
      [this, calls](const batching::GenerationUpdate&, const RequestHandle&) {
        ++*calls;
        sink_gate.arrive();
        throw 1;
      };
  auto handle = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  // A moved-from std::function may still retain its small captures.
  input = {};
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_TRUE(executor.subsequent.wait_for());
  // The sink failure creates a synthetic terminal, but the in-flight engine
  // step is still held and must settle before releasing the session/admission.
  sink_gate.release();
  ASSERT_TRUE(wait_until([&] { return handle.error().has_value(); }));
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  EXPECT_FALSE(handle.done());

  other_sink_gate.hold();
  auto busy_events = std::make_shared<Events>();
  auto busy_input = request("session", busy_events, 1);
  busy_input.on_update = [this, busy_events](
                             const batching::GenerationUpdate& update,
                             const RequestHandle&) {
    busy_events->record(update);
    other_sink_gate.arrive();
  };
  auto busy =
      accepted(GenerationBridge::submit(*runtime, std::move(busy_input)));
  ASSERT_TRUE(other_sink_gate.wait_for());
  ASSERT_TRUE(busy.error());
  EXPECT_EQ(busy.error()->code, ErrorCode::SessionBusy);
  EXPECT_FALSE(busy.done());
  // The selected busy terminal released its slot, but the original request
  // still owns the session. At most R registry entries plus this callback live.
  auto another_busy_events = std::make_shared<Events>();
  auto another_busy = accepted(GenerationBridge::submit(
      *runtime, request("session", another_busy_events, 1)));
  opened("session"); // Control rejects the newly admitted request as busy.
  ASSERT_TRUE(another_busy.error());
  EXPECT_EQ(another_busy.error()->code, ErrorCode::SessionBusy);
  EXPECT_FALSE(another_busy.done());
  auto rejected_events = std::make_shared<Events>();
  auto rejected = GenerationBridge::submit(
      *runtime, request("session", rejected_events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  EXPECT_TRUE(rejected_events->updates.empty());
  other_sink_gate.release();
  completed(busy);
  completed(another_busy);
  expect_terminal(busy_events, batching::FinishReason::Failed);
  expect_terminal(another_busy_events, batching::FinishReason::Failed);
  EXPECT_FALSE(handle.done());
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(executor.calls.load(), 2);
  {
    std::lock_guard<std::mutex> lock(events->mutex);
    EXPECT_FALSE(events->prepared);
    EXPECT_FALSE(events->completion);
  }

  executor.subsequent.release();
  completed(handle);
  EXPECT_EQ(calls->load(), 1);
  EXPECT_EQ(handle.error()->code, ErrorCode::Internal);
  {
    std::lock_guard<std::mutex> lock(events->mutex);
    ASSERT_TRUE(events->completion);
    ASSERT_TRUE(events->completion->error);
    EXPECT_EQ(events->completion->error->code, ErrorCode::Internal);
    EXPECT_EQ(
        events->completion->terminal.finish_reason,
        batching::FinishReason::Failed);
    EXPECT_TRUE(events->completion->current_session);
    EXPECT_GE(events->completion->metrics.n_generated_tokens, 1);
    EXPECT_TRUE(events->updates.empty());
  }
  std::weak_ptr<Events> capture = events;
  events.reset();
  EXPECT_TRUE(capture.expired());
  std::weak_ptr<std::atomic<int>> sink_capture = calls;
  calls.reset();
  EXPECT_TRUE(sink_capture.expired());
  auto next_events = std::make_shared<Events>();
  auto next = accepted(
      GenerationBridge::submit(*runtime, request("session", next_events, 1)));
  completed(next);
  EXPECT_FALSE(next.error());
  expect_terminal(next_events, batching::FinishReason::NewTokenLimit);
}
#endif

TEST_F(GenerationBridgeTest, InvalidSubmissionAndDefaultHandleHaveNoCallbacks) {
  RequestHandle empty;
  EXPECT_EQ(empty.id(), 0u);
  EXPECT_FALSE(empty.done());
  EXPECT_FALSE(empty.error());
  empty.cancel();
  empty.wait();
  start();
  auto events = std::make_shared<Events>();
  auto input = request("session", events);
  input.delta = std::vector<batching::Token>{};
  auto result = GenerationBridge::submit(*runtime, std::move(input));
  ASSERT_TRUE(std::holds_alternative<ServingError>(result));
  EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
  EXPECT_TRUE(events->updates.empty());
  EXPECT_TRUE(executor.opened().empty());
}

} // namespace

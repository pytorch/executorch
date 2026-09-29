/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/serving/detail/generation_bridge.h>
#include <executorch/extension/llm/serving/serving_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <random>
#include <set>
#include <thread>

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/test/fake_executor.h>
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
  bool wait_for(std::size_t count = 1) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, kTimeout, [&] { return arrivals_ >= count; });
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
  Gate initializing;
  Gate opening;
  Gate executing;
  Gate subsequent;
  std::atomic<int> calls{0};
  std::atomic<int> clones{0};
  std::atomic<bool> shared_batch{false};
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
    result.delta = {10, 11};
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
    runtime.reset();
    EXPECT_EQ(executor.clones.load(), 0);
  }
  Executor executor;
  Gate sink_gate;
  Gate other_sink_gate;
  std::unique_ptr<ServingRuntime> runtime;
};

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

TEST_F(GenerationBridgeTest, ResetAndCloseFenceStaleCompletionFromReplacement) {
  executor.subsequent.hold();
  start();
  sink_gate.hold();
  auto old_events = std::make_shared<Events>();
  auto input = request("session", old_events, 2);
  input.on_update = [this, old_events](
                        const batching::GenerationUpdate& update,
                        const RequestHandle&) {
    if (!update.finish_reason) {
      sink_gate.arrive();
    }
    old_events->record(update);
  };
  auto old = accepted(GenerationBridge::submit(*runtime, std::move(input)));
  ASSERT_TRUE(sink_gate.wait_for());
  ASSERT_TRUE(executor.subsequent.wait_for());
  auto reset = runtime->reset_session_async("session");
  executor.subsequent.release();
  ASSERT_EQ(reset.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(reset.get());
  auto replacement_events = std::make_shared<Events>();
  auto replacement = accepted(GenerationBridge::submit(
      *runtime, request("session", replacement_events, 1)));
  opened("session"); // Control has bound the replacement before old delivery.
  sink_gate.release();
  completed(replacement);
  completed(old);
  {
    std::lock_guard<std::mutex> lock(old_events->mutex);
    ASSERT_TRUE(old_events->completion);
    EXPECT_FALSE(old_events->completion->current_session);
    EXPECT_FALSE(old_events->completion->position);
  }
  EXPECT_EQ(runtime->info().active_sessions, 1u);
  EXPECT_EQ(executor.opened().size(), 2u);

  executor.executing.hold();
  const auto next_call = executor.calls.load() + 1;
  auto closing_events = std::make_shared<Events>();
  auto closing = accepted(GenerationBridge::submit(
      *runtime, request("session", closing_events, 5)));
  ASSERT_TRUE(executor.executing.wait_for(next_call));
  auto closed = runtime->close_session_async("session");
  ASSERT_EQ(closed.wait_for(kTimeout), std::future_status::ready);
  EXPECT_FALSE(closed.get());
  EXPECT_EQ(runtime->info().active_sessions, 0u);
  executor.executing.release();
  completed(closing);
  expect_terminal(closing_events, batching::FinishReason::Cancelled);
  std::lock_guard<std::mutex> lock(closing_events->mutex);
  EXPECT_FALSE(closing_events->completion->current_session);
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
  auto rejected_events = std::make_shared<Events>();
  auto rejected = GenerationBridge::submit(
      *runtime, request("session", rejected_events, 1));
  ASSERT_TRUE(std::holds_alternative<ServingError>(rejected));
  EXPECT_EQ(std::get<ServingError>(rejected).code, ErrorCode::CapacityExceeded);
  EXPECT_TRUE(rejected_events->updates.empty());
  other_sink_gate.release();
  completed(busy);
  expect_terminal(busy_events, batching::FinishReason::Failed);
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
  input.delta.clear();
  auto result = GenerationBridge::submit(*runtime, std::move(input));
  ASSERT_TRUE(std::holds_alternative<ServingError>(result));
  EXPECT_EQ(std::get<ServingError>(result).code, ErrorCode::InvalidArgument);
  EXPECT_TRUE(events->updates.empty());
  EXPECT_TRUE(executor.opened().empty());
}

} // namespace

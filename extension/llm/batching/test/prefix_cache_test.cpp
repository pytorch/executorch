/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/decode_first_scheduler.h>
#include <executorch/extension/llm/batching/prefix_cache.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace executorch::extension::llm::batching;
using Tokens = std::vector<Token>;
constexpr std::chrono::seconds kTimeout{5};

// Predictions depend on the whole committed history. A bad restore therefore
// changes outputs even if the runner submits the expected suffix and position.
class HistoryExecutor final : public Executor {
 public:
  struct Seen {
    Position position;
    Tokens tokens;
    SessionId sid;
  };
  struct Clone {
    SessionId source;
    Position position;
    std::optional<SessionId> created;
  };

  std::optional<SessionId> open_session() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (states_.size() >= max_sessions.load()) {
      return std::nullopt;
    }
    const auto id = next_id_++;
    states_.emplace(id, State{});
    return id;
  }

  void close_session(SessionId id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    EXPECT_EQ(states_.erase(id), 1u);
    closed_.push_back(id);
    cv_.notify_all();
  }

  std::optional<SessionId> clone(SessionId source, Position upto) override {
    std::lock_guard<std::mutex> lock(mutex_);
    clones_.push_back(Clone{source, upto, std::nullopt});
#if ET_HAS_EXCEPTIONS
    if (throw_next_clone.exchange(false)) {
      throw std::bad_alloc();
    }
#endif
    const auto it = states_.find(source);
    if (it == states_.end() || upto <= 0 ||
        static_cast<std::size_t>(upto) > it->second.tokens.size() ||
        upto <
            std::max(Position{0}, it->second.written - retained_tail.load()) ||
        states_.size() >= max_sessions.load() || reject_clones.load()) {
      return std::nullopt;
    }
    State copy = it->second;
    copy.tokens.resize(static_cast<std::size_t>(upto));
    copy.seed.reset();
    const auto id = next_id_++;
    states_.emplace(id, std::move(copy));
    clones_.back().created = id;
    return id;
  }

  void set_sampling(
      SessionId id,
      const SamplingParams&,
      std::optional<std::uint64_t> seed) override {
    std::lock_guard<std::mutex> lock(mutex_);
    states_.at(id).seed = seed;
  }

  bool execute(const BatchInput& batch, BatchOutput& out) override {
    std::lock_guard<std::mutex> lock(mutex_);
    out.outputs.assign(batch.inputs.size(), std::nullopt);
    for (std::size_t i = 0; i < batch.inputs.size(); ++i) {
      const auto& input = batch.inputs[i];
      auto& state = states_.at(input.sid);
      const auto start = input.position + static_cast<Position>(input.offset);
      if (start < 0 || static_cast<std::size_t>(start) > state.tokens.size() ||
          start < std::max(Position{0}, state.written - retained_tail.load())) {
        return false;
      }
      state.tokens.resize(static_cast<std::size_t>(start));
      Tokens supplied(
          std::get<TokenInputPtr>(input.payload)->begin() + input.offset,
          std::get<TokenInputPtr>(input.payload)->begin() + input.offset +
              input.size);
      state.tokens.insert(state.tokens.end(), supplied.begin(), supplied.end());
      state.written =
          std::max(state.written, static_cast<Position>(state.tokens.size()));
      seen_.push_back(Seen{start, std::move(supplied), input.sid});
      if (input.produce_output) {
        Token prediction = 1469598103934665603ULL ^ state.seed.value_or(0);
        for (Token token : state.tokens) {
          prediction = (prediction ^ token) * 1099511628211ULL;
        }
        out.outputs[i] = Output{input.sid, {prediction}};
      }
    }
    return true;
  }

  std::vector<Seen> seen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_;
  }
  void clear_seen() {
    std::lock_guard<std::mutex> lock(mutex_);
    seen_.clear();
  }
  std::vector<Clone> clones() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return clones_;
  }
  std::vector<SessionId> closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }
  bool wait_for_live(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(
        lock, kTimeout, [&] { return states_.size() == count; });
  }

  std::atomic<std::size_t> max_sessions{64};
  std::atomic<Position> retained_tail{std::numeric_limits<Position>::max()};
  std::atomic<bool> reject_clones{false};
  std::atomic<bool> throw_next_clone{false};

 private:
  struct State {
    Tokens tokens;
    Position written = 0;
    std::optional<std::uint64_t> seed;
  };
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  SessionId next_id_ = 0;
  std::map<SessionId, State> states_;
  std::vector<Seen> seen_;
  std::vector<Clone> clones_;
  std::vector<SessionId> closed_;
};

class PrefixCacheTest : public ::testing::Test {
 protected:
  struct Generation {
    Tokens tokens;
    std::optional<Session> snapshot;
  };

  PrefixCacheTest()
      : runner(std::make_unique<Runner>(
            executor,
            DecodeFirstScheduler::create(16, 4, 4))) {}

  Session open() {
    auto future = runner->open_session_async();
    if (future.wait_for(kTimeout) != std::future_status::ready) {
      ADD_FAILURE() << "open_session_async did not settle within the timeout";
      return {};
    }
    auto session = future.get();
    EXPECT_TRUE(session);
    return session ? std::move(*session) : Session{};
  }

  Generation generate(
      const Session& session,
      const Tokens& delta,
      int count = 4,
      std::uint64_t seed = 17,
      std::optional<Position> capture = std::nullopt) {
    Generation result;
    std::future<std::optional<Session>> snapshot;
    std::promise<void> terminal;
    auto done = terminal.get_future();
    GenConfig config;
    config.max_new_tokens = count;
    config.seed = seed;
    auto handle =
        session.generate_async(delta, config, [&](const auto& update) {
          if (capture && !snapshot.valid() && !update.tokens.empty()) {
            snapshot = session.clone_async(*capture);
          }
          result.tokens.insert(
              result.tokens.end(), update.tokens.begin(), update.tokens.end());
          if (update.finish_reason) {
            terminal.set_value();
          }
        });
    const auto ready = done.wait_for(kTimeout);
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) {
      handle.cancel();
    }
    handle.wait();
    EXPECT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit)
        << handle.error_message();
    if (snapshot.valid()) {
      if (snapshot.wait_for(kTimeout) == std::future_status::ready) {
        result.snapshot = snapshot.get();
      } else {
        ADD_FAILURE() << "clone_async did not settle within the timeout";
      }
    }
    return result;
  }

  GenerationHandle run_callback(
      const Session& session,
      const Tokens& delta,
      GenerationCallback callback,
      int count = 4,
      const Tokens& stop_tokens = {}) {
    std::promise<void> terminal;
    auto done = terminal.get_future();
    GenConfig config;
    config.max_new_tokens = count;
    config.seed = 17;
    config.stop_tokens = stop_tokens;
    auto handle = session.generate_async(
        delta,
        config,
        [callback = std::move(callback), &terminal](const auto& update) {
          if (callback) {
            callback(update);
          }
          if (update.finish_reason) {
            terminal.set_value();
          }
        });
    const auto ready = done.wait_for(kTimeout);
    EXPECT_EQ(ready, std::future_status::ready);
    if (ready != std::future_status::ready) {
      handle.cancel();
    }
    handle.wait();
    return handle;
  }

  Session snapshot(const Tokens& prompt) {
    auto source = open();
    auto result =
        generate(source, prompt, 4, 17, static_cast<Position>(prompt.size()));
    EXPECT_TRUE(result.snapshot);
    if (!result.snapshot) {
      return {};
    }
    EXPECT_EQ(result.snapshot->position(), prompt.size());
    return std::move(*result.snapshot);
  }

  void expect_cold_match(PrefixMatch& match, const Tokens& prompt) {
    ASSERT_LT(match.matched_tokens, prompt.size());
    ASSERT_EQ(match.session.position(), match.matched_tokens);
    executor.clear_seen();
    const Tokens suffix(prompt.begin() + match.matched_tokens, prompt.end());
    const auto cached = generate(match.session, suffix, 4, 29);
    const auto seen = executor.seen();
    ASSERT_FALSE(seen.empty());
    EXPECT_EQ(seen.front().position, match.matched_tokens);
    Tokens supplied;
    Position next = static_cast<Position>(match.matched_tokens);
    for (const auto& input : seen) {
      EXPECT_EQ(input.position, next);
      supplied.insert(supplied.end(), input.tokens.begin(), input.tokens.end());
      next += static_cast<Position>(input.tokens.size());
    }
    ASSERT_GE(supplied.size(), suffix.size());
    EXPECT_EQ(
        Tokens(supplied.begin(), supplied.begin() + suffix.size()), suffix);
    EXPECT_EQ(match.session.position(), prompt.size() + 3);
    auto cold = open();
    EXPECT_EQ(cached.tokens, generate(cold, prompt, 4, 29).tokens);
  }

  HistoryExecutor executor;
  std::unique_ptr<Runner> runner;
};

class PrefillBreakpointTest
    : public PrefixCacheTest,
      public ::testing::WithParamInterface<std::size_t> {};

TEST_P(PrefillBreakpointTest, ClonesExactPositionsBeforeLaterWrites) {
  runner.reset();
  runner = std::make_unique<Runner>(
      executor, DecodeFirstScheduler::create(64, 4, GetParam()));
  executor.retained_tail = 0;
  auto source = open();
  const Tokens prompt(40, 7);
  std::vector<Position> positions;
  std::vector<std::future<std::optional<Session>>> snapshots;
  GenConfig config;
  config.max_new_tokens = 4;
  config.prefill_breakpoints = {5, 17, 31, 40};
  auto generation =
      source.generate_async(prompt, config, {}, {}, [&](Position position) {
        EXPECT_EQ(position, source.position());
        positions.push_back(position);
        snapshots.push_back(source.clone_async(position));
      });
  generation.wait();
  ASSERT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_EQ(positions, config.prefill_breakpoints);
  EXPECT_FALSE(source.clone_async(positions.front()).get());
  PrefixCache cache(1);
  for (std::size_t i = 0; i < snapshots.size(); ++i) {
    auto snapshot = snapshots[i].get();
    ASSERT_TRUE(snapshot);
    EXPECT_EQ(snapshot->position(), positions[i]);
    if (i == 0) {
      ASSERT_TRUE(cache.insert(
          Tokens(prompt.begin(), prompt.begin() + positions[i]),
          std::move(*snapshot)));
    }
  }
  Tokens changed(prompt.begin(), prompt.begin() + positions.front());
  changed.push_back(9);
  auto match = cache.lookup(changed);
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, positions.front());
  expect_cold_match(*match, changed);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkSizes,
    PrefillBreakpointTest,
    ::testing::Values(4u, 13u, 40u));

TEST_F(PrefixCacheTest, ContinuationBreakpointsIncludeThePendingToken) {
  executor.retained_tail = 0;
  auto source = open();
  const Tokens initial(5, 3);
  const auto first = generate(source, initial, 1);
  ASSERT_EQ(source.position(), 5);
  const Tokens delta(19, 7);
  GenConfig config;
  config.max_new_tokens = 2;
  config.seed = 17;
  config.prefill_breakpoints = periodic_prefill_breakpoints(5, 25, 8, true);
  std::vector<Position> positions;
  std::vector<std::future<std::optional<Session>>> snapshots;
  Tokens output;
  auto handle = source.generate_async(
      delta,
      config,
      [&](const GenerationUpdate& update) {
        output.insert(output.end(), update.tokens.begin(), update.tokens.end());
      },
      {},
      [&](Position position) {
        positions.push_back(position);
        snapshots.push_back(source.clone_async(position));
      });
  handle.wait();
  ASSERT_EQ(handle.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(positions, (std::vector<Position>{8, 16, 24, 25}));
  for (auto& snapshot : snapshots) {
    EXPECT_TRUE(snapshot.get());
  }
  Tokens full = initial;
  full.insert(full.end(), first.tokens.begin(), first.tokens.end());
  full.insert(full.end(), delta.begin(), delta.end());
  auto cold = open();
  EXPECT_EQ(output, generate(cold, full, 2).tokens);
}

TEST_F(PrefixCacheTest, InvalidBreakpointsFailBeforePrefill) {
  auto source = open();
  for (const std::vector<Position>& positions :
       {std::vector<Position>{0}, {9}, {3, 3}, {4, 2}}) {
    GenConfig config;
    config.prefill_breakpoints = positions;
    auto handle = source.generate_async(Tokens(8, 7), config, {});
    handle.wait();
    EXPECT_EQ(handle.finish_reason(), FinishReason::Failed);
    EXPECT_EQ(
        handle.error_message(),
        "prefill breakpoints must increase within the generation's prefill range");
    EXPECT_EQ(source.position(), 0);
  }
  EXPECT_TRUE(executor.seen().empty());
}

TEST_F(PrefixCacheTest, ClosingAtBreakpointDropsDeferredPrefill) {
  auto source = open();
  GenConfig config;
  config.prefill_breakpoints = {5, 17};
  auto handle = source.generate_async(
      Tokens(40, 7), config, {}, {}, [&](Position position) {
        EXPECT_EQ(position, 5);
        source = Session{};
      });
  handle.wait();
  EXPECT_EQ(handle.finish_reason(), FinishReason::Cancelled);
  const auto seen = executor.seen();
  ASSERT_FALSE(seen.empty());
  EXPECT_EQ(seen.back().position + seen.back().tokens.size(), 5u);
}

TEST(PrefillCheckpointPolicyTest, AbsoluteIntervalsAndRangeBoundaries) {
  EXPECT_EQ(
      periodic_prefill_breakpoints(0, 24, 8),
      (std::vector<Position>{8, 16, 24}));
  EXPECT_EQ(
      periodic_prefill_breakpoints(9, 26, 8, true),
      (std::vector<Position>{16, 24, 26}));
  EXPECT_EQ(
      periodic_prefill_breakpoints(16, 17, 8, true),
      (std::vector<Position>{17}));
  EXPECT_TRUE(periodic_prefill_breakpoints(16, 17, 8).empty());
  EXPECT_TRUE(periodic_prefill_breakpoints(0, 8, 0).empty());
  const auto end = std::numeric_limits<Position>::max();
  EXPECT_EQ(
      periodic_prefill_breakpoints(end - 1, end, end, true),
      (std::vector<Position>{end}));
}

TEST_F(PrefixCacheTest, CaptureObserverHonorsAdmissionAndReportsQueueFailure) {
  auto source = open();
  generate(source, Tokens(4, 7), 1);
  PrefixCache cache(1);
  auto capture = cache.capture_prompt(source, Tokens(4, 7));
  bool available = false;
  std::vector<bool> queued;
  auto observer = capture.observer(
      [&] { return available; }, [&](bool value) { queued.push_back(value); });
  observer(4);
  EXPECT_TRUE(queued.empty());
  EXPECT_TRUE(executor.clones().empty());
  available = true;
  observer(5); // Outside the supplied identity.
  observer(4);
  observer(4); // A capture is already pending.
  EXPECT_EQ(queued, (std::vector<bool>{false, true, false}));
  ASSERT_TRUE(capture.collect());
  EXPECT_EQ(executor.clones().size(), 1u);
  executor.reject_clones = true;
  observer(4);
  EXPECT_TRUE(queued.back());
  EXPECT_FALSE(capture.collect());
  executor.reject_clones = false;
  auto match = cache.lookup({7, 7, 7, 7, 9});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, 4u);
}

#if ET_HAS_EXCEPTIONS
TEST_F(PrefixCacheTest, PrefillObserverExceptionFailsWithoutDeliveringTokens) {
  auto source = open();
  Tokens delivered;
  GenConfig config;
  config.prefill_breakpoints = {5, 17};
  auto generation = source.generate_async(
      Tokens(40, 7),
      config,
      [&](const GenerationUpdate& update) {
        delivered.insert(
            delivered.end(), update.tokens.begin(), update.tokens.end());
      },
      {},
      [](Position) { throw std::runtime_error("observer"); });
  generation.wait();
  EXPECT_EQ(generation.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(generation.error_message(), "prefill observer threw");
  EXPECT_TRUE(delivered.empty());
  EXPECT_EQ(source.position(), 5);
}
#endif

TEST_F(
    PrefixCacheTest,
    IntermediateCaptureTruncatesOpaqueIdentityAndCanBeReused) {
  auto source = open();
  const auto tokens = std::make_shared<const Tokens>(Tokens(12, 7));
  const PrefixIdentity identity{
      {TokenSpan{tokens, 0, 2},
       OpaqueSpan{ContentKey{7}, 0, 6},
       TokenSpan{tokens, 8, 4}}};
  PrefixCache cache(2);
  auto capture = cache.capture_identity(source, identity);
  generate(source, Tokens(4, 7), 1);
  ASSERT_TRUE(capture.capture_at(4));
  EXPECT_FALSE(capture.capture_at(4));
  ASSERT_TRUE(capture.collect());
  EXPECT_FALSE(capture.collect());
  auto first = cache.lookup_identity(identity);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->matched_tokens, 4u);
  generate(source, Tokens(5, 7), 1);
  ASSERT_TRUE(capture.capture_at(10));
  ASSERT_TRUE(capture.collect());
  auto second = cache.lookup_identity(identity);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->matched_tokens, 10u);
  EXPECT_FALSE(capture.capture_at(13));
}

TEST_F(PrefixCacheTest, CustomRetentionSelectsVictimFromSnapshotIdentities) {
  executor.retained_tail = 0;
  PrefixCache cache(
      2,
      [](const PrefixCache::EntryView& incoming,
         const PrefixCache::Entries& entries) {
        EXPECT_EQ(incoming.identity.size(), 12u);
        EXPECT_EQ(entries.size(), 2u);
        return std::max_element(
                   entries.begin(),
                   entries.end(),
                   [](const auto& a, const auto& b) {
                     return a.identity.size() < b.identity.size();
                   }) -
            entries.begin();
      });
  const Tokens prompt(16, 7);
  for (std::size_t size : {4, 8, 12}) {
    const Tokens prefix(prompt.begin(), prompt.begin() + size);
    auto source = open();
    generate(source, prefix, 1);
    auto checkpoint = source.clone_async(static_cast<Position>(size)).get();
    ASSERT_TRUE(checkpoint);
    ASSERT_TRUE(cache.insert(prefix, std::move(*checkpoint)));
  }
  EXPECT_EQ(cache.size(), 2u);
  for (std::size_t edit : {4, 8, 12}) {
    Tokens changed(prompt.begin(), prompt.begin() + edit);
    changed.push_back(9);
    auto match = cache.lookup(changed);
    ASSERT_TRUE(match);
    EXPECT_EQ(match->matched_tokens, edit == 12 ? 12u : 4u);
    expect_cold_match(*match, changed);
  }
}

TEST_F(PrefixCacheTest, OneCheckpointEntryStillEvictsAtCapacity) {
  executor.retained_tail = 0;
  PrefixCache cache(1);
  for (std::size_t size : {4, 8}) {
    auto source = open();
    generate(source, Tokens(size, 7), 1);
    auto checkpoint = source.clone_async(static_cast<Position>(size)).get();
    ASSERT_TRUE(checkpoint);
    ASSERT_TRUE(cache.insert(Tokens(size, 7), std::move(*checkpoint)));
  }
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_FALSE(cache.lookup(Tokens{7, 7, 7, 7, 9}));
  auto match = cache.lookup(Tokens{7, 7, 7, 7, 7, 7, 7, 7, 9});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, 8u);
}

TEST_F(PrefixCacheTest, DeclinedOrInvalidRetentionLeavesSnapshotsUsable) {
  std::optional<std::size_t> victim;
#if ET_HAS_EXCEPTIONS
  bool throw_policy = false;
#endif
  std::size_t calls = 0;
  PrefixCache cache(2, [&](const auto&, const auto& entries) {
    ++calls;
    EXPECT_EQ(entries.size(), 2u);
#if ET_HAS_EXCEPTIONS
    if (throw_policy) {
      throw std::runtime_error("retention failed");
    }
#endif
    return victim;
  });
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2})));
  ASSERT_TRUE(cache.insert({3, 4}, snapshot({3, 4})));
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2})));
  EXPECT_EQ(calls, 0u);
  auto hints = std::make_shared<const int>(42);
  std::weak_ptr<const int> declined = hints;
  EXPECT_FALSE(cache.insert({5, 6}, snapshot({5, 6}), std::move(hints)));
  EXPECT_TRUE(declined.expired());
  victim = 2;
  EXPECT_FALSE(cache.insert({5, 6}, snapshot({5, 6})));
#if ET_HAS_EXCEPTIONS
  throw_policy = true;
  EXPECT_FALSE(cache.insert({5, 6}, snapshot({5, 6})));
#endif
  EXPECT_EQ(cache.size(), 2u);
  ASSERT_TRUE(executor.wait_for_live(2));
  EXPECT_FALSE(cache.lookup({5, 6, 9}));
  for (const Tokens prompt : {Tokens{1, 2, 9}, Tokens{3, 4, 9}}) {
    auto match = cache.lookup(prompt);
    ASSERT_TRUE(match);
    expect_cold_match(*match, prompt);
  }
}

TEST_F(PrefixCacheTest, MetadataCanProtectAnAnchorWithoutChangingMatching) {
  PrefixCache cache(2, [](const auto& incoming, const auto& entries) {
    EXPECT_EQ(*std::static_pointer_cast<const int>(incoming.metadata), 2);
    return std::min_element(
               entries.begin(),
               entries.end(),
               [](const auto& a, const auto& b) {
                 return *std::static_pointer_cast<const int>(a.metadata) <
                     *std::static_pointer_cast<const int>(b.metadata);
               }) -
        entries.begin();
  });
  auto anchor = std::make_shared<const int>(3);
  auto expendable = std::make_shared<const int>(1);
  std::weak_ptr<const int> evicted = expendable;
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2}), anchor));
  ASSERT_TRUE(cache.insert({3, 4}, snapshot({3, 4}), std::move(expendable)));
  ASSERT_FALSE(evicted.expired());
  ASSERT_TRUE(
      cache.insert({5, 6}, snapshot({5, 6}), std::make_shared<const int>(2)));
  EXPECT_TRUE(evicted.expired());
  EXPECT_FALSE(cache.lookup({3, 4, 9}));
  auto match = cache.lookup({1, 2, 9});
  ASSERT_TRUE(match);
  expect_cold_match(*match, {1, 2, 9});
}

TEST_F(PrefixCacheTest, DuplicateRefreshesMetadataAndReleasesPreviousHints) {
  auto original = std::make_shared<const int>(1);
  auto refreshed = std::make_shared<const int>(2);
  std::weak_ptr<const int> old = original;
  std::weak_ptr<const int> current = refreshed;
  std::size_t calls = 0;
  PrefixCache cache(1, [&](const auto& incoming, const auto& entries) {
    ++calls;
    EXPECT_FALSE(incoming.metadata);
    EXPECT_EQ(entries.front().metadata, current.lock());
    return 0;
  });
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2}), std::move(original)));
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2}), std::move(refreshed)));
  EXPECT_EQ(calls, 0u);
  EXPECT_TRUE(old.expired());
  EXPECT_FALSE(current.expired());
  ASSERT_TRUE(cache.lookup({1, 2, 9}));
  ASSERT_TRUE(cache.insert({3, 4}, snapshot({3, 4})));
  EXPECT_EQ(calls, 1u);
  EXPECT_TRUE(current.expired());
}

TEST_F(PrefixCacheTest, CaptureKeepsMetadataWithItsPendingSnapshot) {
  auto source = open();
  generate(source, Tokens(12, 7), 1);
  auto first = std::make_shared<const int>(4);
  auto second = std::make_shared<const int>(8);
  std::size_t calls = 0;
  PrefixCache cache(1, [&](const auto& incoming, const auto& entries) {
    ++calls;
    EXPECT_EQ(incoming.identity.size(), 8u);
    EXPECT_EQ(incoming.metadata, second);
    EXPECT_EQ(entries.front().identity.size(), 4u);
    EXPECT_EQ(entries.front().metadata, first);
    return 0;
  });
  auto capture = cache.capture_prompt(source, Tokens(12, 7));
  ASSERT_TRUE(capture.capture_at(4, first));
  EXPECT_FALSE(capture.capture_at(8, second));
  ASSERT_TRUE(capture.collect());
  ASSERT_TRUE(capture.capture_at(8, second));
  ASSERT_TRUE(capture.collect());
  EXPECT_EQ(calls, 1u);
  executor.reject_clones = true;
  auto refused = std::make_shared<const int>(12);
  std::weak_ptr<const int> discarded = refused;
  ASSERT_TRUE(capture.capture_at(12, std::move(refused)));
  EXPECT_FALSE(capture.collect());
  EXPECT_TRUE(discarded.expired());
  EXPECT_EQ(calls, 1u);
}

TEST_F(PrefixCacheTest, WrappedCaptureOwnsMetadataUntilCollection) {
  auto source = open();
  auto hints = std::make_shared<const int>(7);
  std::weak_ptr<const int> saved = hints;
  PrefixCache cache(1, [&](const auto&, const auto& entries) {
    EXPECT_EQ(entries.front().metadata, saved.lock());
    EXPECT_FALSE(saved.expired());
    return 0;
  });
  auto capture = cache.capture_prompt(source, {1, 2, 3}, std::move(hints));
  auto generation = run_callback(source, {1, 2, 3}, capture.wrap());
  source = Session{};
  ASSERT_TRUE(capture.collect());
  capture = {};
  EXPECT_FALSE(saved.expired());
  ASSERT_TRUE(cache.insert({8, 8}, snapshot({8, 8})));
  EXPECT_TRUE(saved.expired());
}

TEST_F(PrefixCacheTest, ObserverMatchesMetadataAfterSkippingABoundary) {
  auto source = open();
  generate(source, Tokens(8, 7), 1);
  auto early = std::make_shared<const int>(4);
  auto later = std::make_shared<const int>(8);
  std::size_t calls = 0;
  PrefixCache cache(1, [&](const auto& incoming, const auto& entries) {
    ++calls;
    EXPECT_EQ(incoming.identity.size(), 8u);
    EXPECT_EQ(incoming.metadata, later);
    EXPECT_FALSE(entries.front().metadata);
    return 0;
  });
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2})));
  auto capture = cache.capture_prompt(source, Tokens(8, 7));
  bool available = false;
  std::vector<bool> queued;
  auto observer = capture.observer(
      [&] { return available; },
      [&](bool value) { queued.push_back(value); },
      {{4, early}, {8, later}});
  observer(4);
  EXPECT_TRUE(queued.empty());
  available = true;
  observer(8);
  ASSERT_EQ(queued, (std::vector<bool>{true}));
  ASSERT_TRUE(capture.collect());
  EXPECT_EQ(calls, 1u);
}

TEST_F(PrefixCacheTest, MixedIdentityComparisonIgnoresSpanPartitioning) {
  const auto tokens = std::make_shared<const Tokens>(Tokens{1, 2, 3});
  const ContentKey image{7};
  const PrefixIdentity full{
      {TokenSpan{tokens, 0, 2},
       OpaqueSpan{image, 0, 4},
       TokenSpan{tokens, 2, 1}}};
  const PrefixIdentity split{
      {TokenSpan{tokens, 0, 1},
       TokenSpan{tokens, 1, 1},
       OpaqueSpan{image, 0, 2},
       OpaqueSpan{image, 2, 2},
       TokenSpan{tokens, 2, 1}}};
  const auto max = std::numeric_limits<std::size_t>::max();
  for (const auto& item : std::vector<std::pair<PrefixIdentity, std::size_t>>{
           {split, 7},
           {{{TokenSpan{tokens, 0, 2}, OpaqueSpan{ContentKey{8}, 0, 4}}}, 2},
           {{{TokenSpan{tokens, 0, 2}, OpaqueSpan{image, 1, 4}}}, 2},
           {{{TokenSpan{tokens, 0, 2}, OpaqueSpan{image, 0, 2}}}, 4},
           {{{TokenSpan{tokens, 0, 3}}}, 2},
           {{{TokenSpan{nullptr, 0, 1}}}, 0},
           {{{TokenSpan{tokens, 2, 2}}}, 0},
           {{{OpaqueSpan{image, max, 1}}}, 0},
           {{{OpaqueSpan{image, 0, max}, OpaqueSpan{image, 0, 1}}}, 0},
           {{{TokenSpan{tokens, 0, 0}}}, 0},
           {{}, 0}}) {
    EXPECT_EQ(common_prefix(full, item.first), item.second);
    EXPECT_EQ(common_prefix(item.first, full), item.second);
  }
  PrefixCache cache(1);
  ASSERT_TRUE(cache.insert_identity(full, snapshot({1, 2, 7, 7, 7, 7, 3})));
  ASSERT_TRUE(cache.insert_identity(split, snapshot({1, 2, 7, 7, 7, 7, 3})));
  EXPECT_EQ(cache.size(), 1u);
  auto hit = cache.lookup_identity(split);
  ASSERT_TRUE(hit);
  EXPECT_EQ(hit->matched_tokens, 6u);
}

TEST_F(PrefixCacheTest, EqualShorterAndExtendedPromptsReplayTheFinalToken) {
  PrefixCache cache(2);
  const Tokens prompt{1, 2, 3, 4, 5, 6};
  ASSERT_TRUE(cache.insert(prompt, snapshot(prompt)));
  for (const auto& request :
       std::vector<Tokens>{prompt, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7, 8}}) {
    auto match = cache.lookup(request);
    ASSERT_TRUE(match);
    EXPECT_EQ(
        match->matched_tokens, std::min(prompt.size(), request.size() - 1));
    expect_cold_match(*match, request);
  }
  EXPECT_FALSE(cache.lookup({}));
  EXPECT_FALSE(cache.lookup({1}));
  EXPECT_FALSE(cache.lookup({9, 2, 3}));
}

TEST_F(PrefixCacheTest, BranchesSurviveSourceClosureAndSnapshotEviction) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  ASSERT_TRUE(cache.insert(prompt, snapshot(prompt)));
  ASSERT_TRUE(executor.wait_for_live(1));
  auto left = cache.lookup({1, 2, 3, 8});
  auto right = cache.lookup({1, 2, 3, 9});
  ASSERT_TRUE(left && right);
  ASSERT_TRUE(cache.insert({20, 21}, snapshot({20, 21})));
  cache.clear();
  ASSERT_TRUE(executor.wait_for_live(2));
  expect_cold_match(*left, {1, 2, 3, 8});
  expect_cold_match(*right, {1, 2, 3, 9});
}

TEST_F(PrefixCacheTest, LongestPrefixWinsOverMoreRecentShorterMatch) {
  PrefixCache cache(2);
  ASSERT_TRUE(cache.insert({1, 2, 3, 4, 5}, snapshot({1, 2, 3, 4, 5})));
  const auto long_id = executor.clones().back().created;
  ASSERT_TRUE(cache.insert({1, 2, 9}, snapshot({1, 2, 9})));
  auto match = cache.lookup({1, 2, 3, 4, 8});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, 4u);
  EXPECT_EQ(executor.clones().back().source, long_id);
}

TEST_F(PrefixCacheTest, EqualLengthMatchesPreferTheMostRecentlyUsedSnapshot) {
  PrefixCache cache(2);
  ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
  const auto first_id = executor.clones().back().created;
  ASSERT_TRUE(cache.insert({1, 2, 4}, snapshot({1, 2, 4})));
  const auto second_id = executor.clones().back().created;
  auto tied = cache.lookup({1, 2, 9});
  ASSERT_TRUE(tied);
  EXPECT_EQ(tied->matched_tokens, 2u);
  EXPECT_EQ(executor.clones().back().source, second_id);
  auto exact_first = cache.lookup({1, 2, 3, 8});
  ASSERT_TRUE(exact_first);
  auto next_tie = cache.lookup({1, 2, 9});
  ASSERT_TRUE(next_tie);
  EXPECT_EQ(executor.clones().back().source, first_id);
}

TEST_F(PrefixCacheTest, DuplicateInsertRefreshesLruAndReleasesTheNewSnapshot) {
  PrefixCache cache(2);
  ASSERT_TRUE(cache.insert({1, 2}, snapshot({1, 2})));
  ASSERT_TRUE(cache.insert({3, 4}, snapshot({3, 4})));
  auto duplicate = snapshot({1, 2});
  const auto duplicate_id = executor.clones().back().created;
  ASSERT_TRUE(duplicate_id);
  ASSERT_TRUE(cache.insert({1, 2}, std::move(duplicate)));
  EXPECT_FALSE(duplicate.valid());
  EXPECT_EQ(cache.size(), 2u);
  ASSERT_TRUE(executor.wait_for_live(2));
  const auto closed = executor.closed();
  EXPECT_NE(
      std::find(closed.begin(), closed.end(), *duplicate_id), closed.end());
  ASSERT_TRUE(cache.insert({5, 6}, snapshot({5, 6})));
  EXPECT_FALSE(cache.lookup({3, 4, 9}));
  EXPECT_TRUE(cache.lookup({1, 2, 9}));
}

TEST_F(PrefixCacheTest, SuccessfulLookupRefreshesLruBeforeEviction) {
  for (const auto& policy : std::vector<PrefixCacheRetentionPolicy>{
           {}, lru_prefix_cache_retention}) {
    PrefixCache cache(2, policy);
    ASSERT_TRUE(
        cache.insert({1, 2}, snapshot({1, 2}), std::make_shared<const int>(1)));
    ASSERT_TRUE(cache.insert(
        {3, 4}, snapshot({3, 4}), std::make_shared<const int>(100)));
    EXPECT_TRUE(cache.lookup({1, 2, 9}));
    ASSERT_TRUE(cache.insert({5, 6}, snapshot({5, 6})));
    EXPECT_FALSE(cache.lookup({3, 4, 9}));
    EXPECT_TRUE(cache.lookup({1, 2, 9}));
  }
}

TEST_F(PrefixCacheTest, TokenKeysPreserveAllSixtyFourBits) {
  PrefixCache cache(1);
  const Token high = (Token{1} << 63) | 1;
  ASSERT_TRUE(cache.insert({high, 2}, snapshot({high, 2})));
  EXPECT_FALSE(cache.lookup({1, 2, 3}));
  auto match = cache.lookup({high, 2, 3});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, 2u);
  expect_cold_match(*match, {high, 2, 3});
}

TEST_F(PrefixCacheTest, DisabledCacheReleasesInsertedSnapshotWithoutCloning) {
  PrefixCache cache(0);
  auto saved = snapshot({1, 2, 3});
  const auto attempts = executor.clones().size();
  EXPECT_FALSE(cache.insert({1, 2, 3}, std::move(saved)));
  EXPECT_FALSE(saved.valid());
  EXPECT_FALSE(cache.lookup({1, 2, 3, 4}));
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_EQ(executor.clones().size(), attempts);
  EXPECT_TRUE(executor.wait_for_live(0));
}

TEST_F(
    PrefixCacheTest,
    InvalidInsertReleasesOwnershipAndPreservesExistingEntry) {
  PrefixCache cache(1);
  ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
  EXPECT_FALSE(cache.insert({1, 2, 3}, Session{}));
  EXPECT_FALSE(cache.insert({}, snapshot({9})));
  EXPECT_FALSE(cache.insert({9, 8}, snapshot({9, 8, 7})));
  EXPECT_FALSE(cache.insert({9, 8, 7}, snapshot({9, 8})));
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_TRUE(executor.wait_for_live(1));
  EXPECT_TRUE(cache.lookup({1, 2, 3, 4}));
}

TEST_F(PrefixCacheTest, RetentionRefusalPreservesSnapshotAndAllowsLaterHit) {
  executor.retained_tail = 2;
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4, 5, 6, 7, 8};
  ASSERT_TRUE(cache.insert(prompt, snapshot(prompt)));
  ASSERT_TRUE(executor.wait_for_live(1));
  EXPECT_FALSE(cache.lookup({1, 2, 3, 4, 9}));
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_TRUE(executor.wait_for_live(1));
  auto request = prompt;
  request.push_back(9);
  auto match = cache.lookup(request);
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, prompt.size());
  expect_cold_match(*match, request);
}

TEST_F(PrefixCacheTest, OlderSnapshotCanSatisfyARefusedNewerMatch) {
  executor.retained_tail = 2;
  PrefixCache cache(2);
  ASSERT_TRUE(cache.insert({1, 2, 3, 4, 5, 6}, snapshot({1, 2, 3, 4, 5, 6})));
  const auto older_id = executor.clones().back().created;
  const Tokens newer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ASSERT_TRUE(cache.insert(newer, snapshot(newer)));
  const auto newer_id = executor.clones().back().created;
  const auto before = executor.clones().size();
  const Tokens branch{1, 2, 3, 4, 5, 99, 100};
  auto match = cache.lookup(branch);
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, 5u);
  const auto attempts = executor.clones();
  ASSERT_EQ(attempts.size(), before + 2);
  EXPECT_EQ(attempts[before].source, newer_id);
  EXPECT_FALSE(attempts[before].created);
  EXPECT_EQ(attempts[before + 1].source, older_id);
  EXPECT_TRUE(attempts[before + 1].created);
  expect_cold_match(*match, branch);
}

TEST_F(PrefixCacheTest, CapacityRefusalDoesNotLeakOrEvictTheSnapshot) {
  executor.max_sessions = 2;
  PrefixCache cache(1);
  ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
  auto occupied = open();
  EXPECT_FALSE(cache.lookup({1, 2, 3, 4}));
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_TRUE(executor.wait_for_live(2));
  occupied = Session{};
  ASSERT_TRUE(executor.wait_for_live(1));
  auto match = cache.lookup({1, 2, 3, 4});
  ASSERT_TRUE(match);
  cache.clear();
  ASSERT_TRUE(executor.wait_for_live(1));
  expect_cold_match(*match, {1, 2, 3, 4});
  match.reset();
  EXPECT_TRUE(executor.wait_for_live(0));
}

#if ET_HAS_EXCEPTIONS
TEST_F(
    PrefixCacheTest,
    CloneAllocationFailureIsAMissAndTheEngineRemainsUsable) {
  PrefixCache cache(1);
  ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
  executor.throw_next_clone = true;
  EXPECT_FALSE(cache.lookup({1, 2, 3, 4}));
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_TRUE(executor.wait_for_live(1));
  auto match = cache.lookup({1, 2, 3, 4});
  ASSERT_TRUE(match);
  expect_cold_match(*match, {1, 2, 3, 4});
}
#endif

TEST_F(PrefixCacheTest, ClearAndDestructionReleaseOnlyOwnedSnapshots) {
  std::optional<PrefixMatch> match;
  {
    PrefixCache cache(2);
    ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
    ASSERT_TRUE(cache.insert({8, 9}, snapshot({8, 9})));
    match = cache.lookup({1, 2, 3, 4});
    ASSERT_TRUE(match);
    cache.clear();
    EXPECT_EQ(cache.size(), 0u);
    ASSERT_TRUE(executor.wait_for_live(1));
    ASSERT_TRUE(cache.insert({5, 6}, snapshot({5, 6})));
  }
  ASSERT_TRUE(executor.wait_for_live(1));
  expect_cold_match(*match, {1, 2, 3, 4});
  match.reset();
  EXPECT_TRUE(executor.wait_for_live(0));
}

TEST_F(PrefixCacheTest, ManagerAndReturnedSessionsMayOutliveRunner) {
  PrefixCache cache(1);
  ASSERT_TRUE(cache.insert({1, 2, 3}, snapshot({1, 2, 3})));
  auto match = cache.lookup({1, 2, 3, 4});
  ASSERT_TRUE(match);
  runner.reset();
  EXPECT_TRUE(executor.wait_for_live(0));
  EXPECT_FALSE(match->session.valid());
  const auto attempts = executor.clones().size();
  EXPECT_FALSE(cache.lookup({1, 2, 3, 4}));
  EXPECT_EQ(executor.clones().size(), attempts);
  cache.clear();
  EXPECT_EQ(cache.size(), 0u);
  match.reset();
}

TEST_F(PrefixCacheTest, PromptCapturePrecedesDecodeRetentionLoss) {
  executor.retained_tail = 2;
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4, 5, 6, 7, 8};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  auto generation = run_callback(
      source,
      prompt,
      capture.wrap([&](const auto& update) {
        received.insert(
            received.end(), update.tokens.begin(), update.tokens.end());
      }),
      8);
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(received.size(), 8u);
  EXPECT_EQ(source.position(), prompt.size() + 7);
  const auto clones = executor.clones();
  ASSERT_EQ(clones.size(), 1u);
  EXPECT_EQ(clones.front().position, prompt.size());
  EXPECT_TRUE(clones.front().created);
  EXPECT_FALSE(source.clone_async(prompt.size()).get());
  ASSERT_TRUE(capture.collect());
  auto match = cache.lookup(prompt);
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, prompt.size() - 1);
  expect_cold_match(*match, prompt);
}

TEST_F(PrefixCacheTest, PromptCaptureCopiesFullHistoryAfterPrefixReuse) {
  PrefixCache cache(2);
  ASSERT_TRUE(cache.insert({1, 2, 3, 4}, snapshot({1, 2, 3, 4})));
  const Tokens prompt{1, 2, 3, 4, 5, 6};
  auto match = cache.lookup(prompt);
  ASSERT_TRUE(match);
  ASSERT_EQ(match->matched_tokens, 4u);
  Tokens history = prompt;
  auto capture = cache.capture_prompt(match->session, history);
  history.assign({99});
  const Tokens suffix(prompt.begin() + match->matched_tokens, prompt.end());
  auto generation = run_callback(match->session, suffix, capture.wrap());
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_TRUE(capture.collect());
  auto extended = prompt;
  extended.push_back(7);
  auto next = cache.lookup(extended);
  ASSERT_TRUE(next);
  EXPECT_EQ(next->matched_tokens, prompt.size());
  expect_cold_match(*next, extended);
}

TEST_F(PrefixCacheTest, PromptCaptureCapacityRefusalPreservesUserTokens) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  Tokens expected;
  {
    auto cold = open();
    expected = generate(cold, prompt).tokens;
  }
  ASSERT_TRUE(executor.wait_for_live(0));
  executor.max_sessions = 1;
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  auto generation =
      run_callback(source, prompt, capture.wrap([&](const auto& update) {
        received.insert(
            received.end(), update.tokens.begin(), update.tokens.end());
      }));
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(received, expected);
  EXPECT_FALSE(capture.collect());
  EXPECT_FALSE(capture.collect());
  EXPECT_EQ(cache.size(), 0u);
  ASSERT_EQ(executor.clones().size(), 1u);
  EXPECT_FALSE(executor.clones().front().created);
  EXPECT_TRUE(executor.wait_for_live(1));
}

#if ET_HAS_EXCEPTIONS
TEST_F(PrefixCacheTest, PromptCaptureAllocationFailurePreservesUserTokens) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  executor.throw_next_clone = true;
  Tokens received;
  auto generation =
      run_callback(source, prompt, capture.wrap([&](const auto& update) {
        received.insert(
            received.end(), update.tokens.begin(), update.tokens.end());
      }));
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_FALSE(capture.collect());
  EXPECT_EQ(cache.size(), 0u);
  ASSERT_EQ(executor.clones().size(), 1u);
  auto cold = open();
  EXPECT_EQ(received, generate(cold, prompt).tokens);
}

TEST_F(PrefixCacheTest, PromptCaptureDoesNotSwallowUserCallbackException) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  std::promise<void> entered;
  auto called = entered.get_future();
  std::atomic<std::size_t> callbacks{0};
  GenConfig config;
  config.max_new_tokens = 4;
  auto generation =
      source.generate_async(prompt, config, capture.wrap([&](const auto&) {
        if (callbacks.fetch_add(1) == 0) {
          entered.set_value();
        }
        throw std::runtime_error("user callback failed");
      }));
  const auto ready = called.wait_for(kTimeout);
  EXPECT_EQ(ready, std::future_status::ready);
  if (ready != std::future_status::ready) {
    generation.cancel();
  }
  generation.wait();
  EXPECT_EQ(generation.finish_reason(), FinishReason::Failed);
  EXPECT_NE(
      generation.error_message().find("user callback failed"),
      std::string::npos);
  capture.collect();
  EXPECT_FALSE(capture.collect());
}
#endif

TEST_F(PrefixCacheTest, RejectedGenerationDoesNotCaptureWithoutOutput) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  std::size_t updates = 0;
  auto generation =
      run_callback(source, {}, capture.wrap([&](const auto& update) {
        ++updates;
        EXPECT_TRUE(update.tokens.empty());
        EXPECT_EQ(update.finish_reason, FinishReason::Failed);
      }));
  EXPECT_EQ(generation.finish_reason(), FinishReason::Failed);
  EXPECT_EQ(updates, 1u);
  EXPECT_FALSE(capture.collect());
  EXPECT_TRUE(executor.clones().empty());
  EXPECT_EQ(cache.size(), 0u);
}

TEST_F(PrefixCacheTest, EmptyUpdatesDoNotConsumeFirstTerminalTokenCapture) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto cold = open();
  const auto expected = generate(cold, prompt, 1).tokens;
  ASSERT_EQ(expected.size(), 1u);
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  std::size_t updates = 0;
  auto callback = capture.wrap([&](const auto& update) {
    ++updates;
    received.insert(received.end(), update.tokens.begin(), update.tokens.end());
  });
  callback(GenerationUpdate{{}, std::nullopt, {}});
  EXPECT_TRUE(executor.clones().empty());
  auto generation = run_callback(source, prompt, callback, 4, expected);
  EXPECT_EQ(generation.finish_reason(), FinishReason::StopToken);
  EXPECT_EQ(updates, 2u);
  EXPECT_EQ(received, expected);
  ASSERT_TRUE(capture.collect());
  ASSERT_EQ(executor.clones().size(), 1u);
  EXPECT_EQ(executor.clones().front().position, prompt.size());
}

TEST_F(PrefixCacheTest, PromptCaptureIsQueuedBeforeUserCallbackClosesSource) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  auto generation =
      run_callback(source, prompt, capture.wrap([&](const auto& update) {
        received.insert(
            received.end(), update.tokens.begin(), update.tokens.end());
        if (!update.tokens.empty()) {
          source = Session{};
        }
      }));
  EXPECT_EQ(generation.finish_reason(), FinishReason::Cancelled);
  EXPECT_EQ(received.size(), 1u);
  ASSERT_TRUE(capture.collect());
  ASSERT_TRUE(executor.wait_for_live(1));
  auto match = cache.lookup({1, 2, 3, 4, 5});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, prompt.size());
  expect_cold_match(*match, {1, 2, 3, 4, 5});
}

TEST_F(PrefixCacheTest, MovedPromptCaptureAndCopiedCallbackCollectOnlyOnce) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  auto callback = capture.wrap([&](const auto& update) {
    received.insert(received.end(), update.tokens.begin(), update.tokens.end());
  });
  auto copied_callback = callback;
  PrefixCache::PromptCapture moved;
  moved = std::move(capture);
  EXPECT_FALSE(capture.collect());
  auto generation = run_callback(source, prompt, copied_callback);
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(received.size(), 4u);
  ASSERT_TRUE(moved.collect());
  EXPECT_FALSE(moved.collect());
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_EQ(executor.clones().size(), 1u);
}

TEST_F(PrefixCacheTest, PromptCaptureSurvivesMovingAndFreeingSourceWrapper) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = std::make_unique<Session>(open());
  auto capture = cache.capture_prompt(*source, prompt);
  auto callback = capture.wrap();
  Session owner = std::move(*source);
  source.reset();
  auto generation = run_callback(owner, prompt, std::move(callback));
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_TRUE(capture.collect());
  const auto seen = executor.seen();
  const auto clones = executor.clones();
  ASSERT_FALSE(seen.empty());
  ASSERT_EQ(clones.size(), 1u);
  EXPECT_EQ(clones.front().source, seen.front().sid);
  EXPECT_EQ(clones.front().position, prompt.size());
  auto match = cache.lookup({1, 2, 3, 4, 5});
  ASSERT_TRUE(match);
  EXPECT_EQ(match->matched_tokens, prompt.size());
  expect_cold_match(*match, {1, 2, 3, 4, 5});
}

TEST_F(
    PrefixCacheTest,
    PromptCaptureKeepsOriginalIdentityAfterSourceReassignment) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Session owner = std::move(source);
  source = open();
  generate(source, {9, 8, 7, 6}, 1);
  const auto replacement_seen = executor.seen();
  ASSERT_FALSE(replacement_seen.empty());
  const auto replacement_id = replacement_seen.front().sid;
  executor.clear_seen();
  auto generation = run_callback(owner, prompt, capture.wrap());
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  ASSERT_TRUE(capture.collect());
  const auto seen = executor.seen();
  const auto clones = executor.clones();
  ASSERT_FALSE(seen.empty());
  ASSERT_EQ(clones.size(), 1u);
  EXPECT_EQ(clones.front().source, seen.front().sid);
  EXPECT_NE(clones.front().source, replacement_id);
  auto match = cache.lookup({1, 2, 3, 4, 5});
  ASSERT_TRUE(match);
  expect_cold_match(*match, {1, 2, 3, 4, 5});
}

TEST_F(
    PrefixCacheTest,
    ClosedSourceCaptureForwardsOutputWithoutRetainingSession) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  auto capture = cache.capture_prompt(source, prompt);
  Tokens received;
  std::optional<FinishReason> finished;
  auto callback = capture.wrap([&](const auto& update) {
    received = update.tokens;
    finished = update.finish_reason;
  });
  source = Session{};
  ASSERT_TRUE(executor.wait_for_live(0));
  const GenerationUpdate update{{101, 102}, FinishReason::StopToken, {}};
  callback(update);
  EXPECT_EQ(received, update.tokens);
  EXPECT_EQ(finished, update.finish_reason);
  EXPECT_FALSE(capture.collect());
  EXPECT_TRUE(executor.clones().empty());
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_TRUE(executor.wait_for_live(0));
}

TEST_F(
    PrefixCacheTest,
    DiscardedPromptCaptureLeavesCallbackUsableAndReleasesClone) {
  PrefixCache cache(1);
  const Tokens prompt{1, 2, 3, 4};
  auto source = open();
  Tokens received;
  GenerationCallback callback;
  {
    auto capture = cache.capture_prompt(source, prompt);
    callback = capture.wrap([&](const auto& update) {
      received.insert(
          received.end(), update.tokens.begin(), update.tokens.end());
    });
  }
  auto generation = run_callback(source, prompt, callback);
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_EQ(received.size(), 4u);
  EXPECT_EQ(cache.size(), 0u);
  const auto clones = executor.clones();
  ASSERT_EQ(clones.size(), 1u);
  ASSERT_TRUE(clones.front().created);
  callback = {};
  generation = {};
  ASSERT_TRUE(executor.wait_for_live(1));
  const auto closed = executor.closed();
  EXPECT_NE(
      std::find(closed.begin(), closed.end(), *clones.front().created),
      closed.end());
}

TEST_F(
    PrefixCacheTest,
    InactivePromptCapturesForwardUpdatesAndAllowEmptyCallback) {
  PrefixCache cache(1);
  PrefixCache disabled(0);
  const Tokens prompt{1, 2, 3};
  auto source = open();
  Session invalid;
  PrefixCache::PromptCapture default_capture;
  auto disabled_capture = disabled.capture_prompt(source, prompt);
  auto empty_capture = cache.capture_prompt(source, {});
  auto invalid_capture = cache.capture_prompt(invalid, prompt);
  const GenerationUpdate update{{42}, FinishReason::NewTokenLimit, {}};
  for (auto* capture :
       {&default_capture,
        &disabled_capture,
        &empty_capture,
        &invalid_capture}) {
    Tokens received;
    capture->wrap([&](const auto& delivered) { received = delivered.tokens; })(
        update);
    EXPECT_EQ(received, update.tokens);
    EXPECT_FALSE(capture->wrap());
    EXPECT_FALSE(capture->collect());
  }
  EXPECT_TRUE(executor.clones().empty());
  auto capture = cache.capture_prompt(source, prompt);
  auto generation = run_callback(source, prompt, capture.wrap(), 1);
  EXPECT_EQ(generation.finish_reason(), FinishReason::NewTokenLimit);
  EXPECT_TRUE(capture.collect());
  EXPECT_EQ(cache.size(), 1u);
}

} // namespace

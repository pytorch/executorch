/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/batching/executor_utils.h>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

namespace cache = executorch::extension::llm::cache;
using namespace executorch::extension::llm::batching;
using executorch::runtime::Error;
using executorch::runtime::LoadBackendOptionsMap;

static_assert(!std::is_copy_constructible_v<SequenceGuard>);
static_assert(!std::is_copy_assignable_v<SequenceGuard>);
static_assert(!std::is_move_constructible_v<SequenceGuard>);
static_assert(!std::is_move_assignable_v<SequenceGuard>);

class FakeBatchControl final : public cache::Cache, public cache::BatchControl {
 public:
  int capacity() const override {
    return 128;
  }

  void clear() override {
    positions.clear();
  }

  bool declare_step(const std::vector<int32_t>&) override {
    return false;
  }

  std::optional<int> max_seqs() const override {
    return std::nullopt;
  }

  std::optional<int32_t> seq_new() override {
    const int32_t seq = next_seq_++;
    positions.emplace(seq, 0);
    return seq;
  }

  std::optional<int32_t> seq_clone(int32_t, std::optional<int>) override {
    return std::nullopt;
  }

  bool seq_rm(int32_t seq) override {
    ++remove_calls;
    return positions.erase(seq) != 0;
  }

  bool rewind(int32_t, int) override {
    return false;
  }

  int pos(int32_t seq) const override {
#if ET_HAS_EXCEPTIONS
    if (throw_on_pos) {
      throw std::bad_alloc();
    }
#endif
    const auto it = positions.find(seq);
    return it == positions.end() ? -1 : it->second;
  }

  std::map<int32_t, int> positions;
  int remove_calls = 0;
  bool throw_on_pos = false;

 protected:
  void* face(cache::FaceId id) override {
    return cache::expose<cache::BatchControl>(this, id);
  }

 private:
  int32_t next_seq_ = 0;
};

struct SessionState {
  int32_t seq;
};

class PublishSequenceTest : public ::testing::Test {
 protected:
  FakeBatchControl control;
  int32_t seq = *control.seq_new();
  SessionId next = 7;
  std::map<SessionId, SessionState> sessions;
  int insert_calls = 0;

  std::optional<SessionId> publish(Position expected = 0) {
    return publish_sequence(
        control, seq, expected, next, [&](SessionId sid, int32_t id) {
          ++insert_calls;
          return sessions.emplace(sid, SessionState{id}).second;
        });
  }
};

TEST(SequenceGuardTest, RemovesUnpublishedSequence) {
  FakeBatchControl control;
  const int32_t seq = *control.seq_new();
  {
    SequenceGuard guard(control, seq);
    EXPECT_EQ(control.remove_calls, 0);
    EXPECT_EQ(control.pos(seq), 0);
  }
  EXPECT_EQ(control.remove_calls, 1);
  EXPECT_TRUE(control.positions.empty());
}

TEST(SequenceGuardTest, ReleaseTransfersOwnership) {
  FakeBatchControl control;
  const int32_t seq = *control.seq_new();
  {
    SequenceGuard guard(control, seq);
    guard.release();
    guard.release();
  }
  EXPECT_EQ(control.remove_calls, 0);
  EXPECT_EQ(control.pos(seq), 0);
}

TEST_F(PublishSequenceTest, PublishesStateAndAdvancesId) {
  control.positions.at(seq) = 12;
  const auto sid = publish(12);
  ASSERT_TRUE(sid);
  EXPECT_EQ(*sid, 7);
  EXPECT_EQ(next, 8);
  ASSERT_EQ(sessions.size(), 1);
  EXPECT_EQ(sessions.at(*sid).seq, seq);
  EXPECT_EQ(control.pos(seq), 12);
  EXPECT_EQ(control.remove_calls, 0);
  EXPECT_EQ(insert_calls, 1);
}

TEST_F(PublishSequenceTest, WrongPositionRemovesOnlyNewSequence) {
  const int32_t other = *control.seq_new();
  control.positions.at(seq) = 4;
  EXPECT_FALSE(publish(3));
  EXPECT_EQ(next, 7);
  EXPECT_EQ(insert_calls, 0);
  EXPECT_TRUE(sessions.empty());
  EXPECT_EQ(control.positions.count(seq), 0);
  EXPECT_EQ(control.pos(other), 0);
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, FailedInsertPreservesId) {
  EXPECT_FALSE(
      publish_sequence(control, seq, 0, next, [&](SessionId sid, int32_t id) {
        ++insert_calls;
        EXPECT_EQ(sid, next);
        EXPECT_EQ(id, seq);
        EXPECT_EQ(control.remove_calls, 0);
        return false;
      }));
  EXPECT_EQ(next, 7);
  EXPECT_EQ(insert_calls, 1);
  EXPECT_TRUE(control.positions.empty());
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, DuplicateLeavesExistingSessionAndSequenceIntact) {
  const int32_t other = *control.seq_new();
  sessions.emplace(next, SessionState{other});
  EXPECT_FALSE(publish());
  EXPECT_EQ(next, 7);
  EXPECT_EQ(insert_calls, 1);
  ASSERT_EQ(sessions.size(), 1);
  EXPECT_EQ(sessions.at(next).seq, other);
  EXPECT_EQ(control.positions.count(seq), 0);
  EXPECT_EQ(control.pos(other), 0);
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, ExhaustedIdRejectsWithoutCallingInsert) {
  next = 0;
  EXPECT_FALSE(publish());
  EXPECT_EQ(next, 0);
  EXPECT_EQ(insert_calls, 0);
  EXPECT_TRUE(sessions.empty());
  EXPECT_TRUE(control.positions.empty());
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, MaximumIdPublishesOnceThenExhausts) {
  next = std::numeric_limits<SessionId>::max();
  const auto sid = publish();
  ASSERT_TRUE(sid);
  EXPECT_EQ(*sid, std::numeric_limits<SessionId>::max());
  EXPECT_EQ(next, 0);
  const int32_t published_seq = seq;
  seq = *control.seq_new();
  EXPECT_FALSE(publish());
  EXPECT_EQ(next, 0);
  EXPECT_EQ(insert_calls, 1);
  ASSERT_EQ(sessions.size(), 1);
  EXPECT_EQ(sessions.at(*sid).seq, published_seq);
  EXPECT_EQ(control.pos(published_seq), 0);
  EXPECT_EQ(control.positions.count(seq), 0);
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, FailedMaximumIdRemainsAvailable) {
  next = std::numeric_limits<SessionId>::max();
  EXPECT_FALSE(publish(1));
  EXPECT_EQ(next, std::numeric_limits<SessionId>::max());
  EXPECT_EQ(control.remove_calls, 1);
  seq = *control.seq_new();
  EXPECT_EQ(publish(), std::numeric_limits<SessionId>::max());
  EXPECT_EQ(next, 0);
}

#if ET_HAS_EXCEPTIONS
TEST_F(PublishSequenceTest, StateConstructionExceptionRemovesSequence) {
  struct ThrowingState {
    explicit ThrowingState(int32_t) {
      throw std::bad_alloc();
    }
  };
  std::map<SessionId, ThrowingState> throwing_sessions;
  EXPECT_THROW(
      publish_sequence(
          control,
          seq,
          0,
          next,
          [&](SessionId sid, int32_t id) {
            ++insert_calls;
            return throwing_sessions.emplace(sid, ThrowingState{id}).second;
          }),
      std::bad_alloc);
  EXPECT_EQ(next, 7);
  EXPECT_EQ(insert_calls, 1);
  EXPECT_TRUE(throwing_sessions.empty());
  EXPECT_TRUE(control.positions.empty());
  EXPECT_EQ(control.remove_calls, 1);
}

TEST_F(PublishSequenceTest, PositionExceptionRemovesSequence) {
  control.throw_on_pos = true;
  EXPECT_THROW(publish(), std::bad_alloc);
  EXPECT_EQ(next, 7);
  EXPECT_EQ(insert_calls, 0);
  EXPECT_TRUE(sessions.empty());
  EXPECT_TRUE(control.positions.empty());
  EXPECT_EQ(control.remove_calls, 1);
}
#endif

// Reads the non-owning option spans during the load, just as backend init does.
struct RecordingLoader {
  Error load_method(
      const std::string& method,
      std::nullptr_t,
      std::nullptr_t,
      const LoadBackendOptionsMap* map) {
    ++calls;
    loaded_method = method;
    EXPECT_NE(map, nullptr);
    if (!map) {
      return Error::Internal;
    }
    EXPECT_EQ(map->size(), 1);
    EXPECT_FALSE(map->has_options("UnrelatedBackend"));
    const auto options = map->get_options(backend.c_str());
    EXPECT_EQ(options.size(), 1);
    if (options.size() != 1) {
      return Error::Internal;
    }
    EXPECT_STREQ(options[0].key, cache::kCacheKeyOption);
    const auto* value = std::get_if<
        std::array<char, executorch::runtime::kMaxOptionValueLength>>(
        &options[0].value);
    EXPECT_NE(value, nullptr);
    if (!value) {
      return Error::Internal;
    }
    key = value->data();
    resolved = cache::CacheRegistry::global().get(key);
    EXPECT_NE(resolved, nullptr);
    return result;
  }

  std::string backend = "RecordingBackend";
  std::string loaded_method;
  std::string key;
  std::shared_ptr<cache::Cache> resolved;
  Error result = Error::Ok;
  int calls = 0;
};

TEST(LoadMethodWithCacheTest, BindsLiveCacheWithoutTakingGuardOwnership) {
  auto control = std::make_shared<FakeBatchControl>();
  RecordingLoader loader;
  {
    const cache::InstallGuard guard(control);
    EXPECT_EQ(
        load_method_with_cache(
            loader, "prefill", loader.backend.c_str(), guard),
        Error::Ok);
    EXPECT_EQ(loader.calls, 1);
    EXPECT_EQ(loader.loaded_method, "prefill");
    EXPECT_EQ(loader.resolved, control);
    EXPECT_EQ(cache::CacheRegistry::global().get(loader.key), control);
  }
  EXPECT_EQ(cache::CacheRegistry::global().get(loader.key), nullptr);
  EXPECT_EQ(loader.resolved, control);
}

TEST(LoadMethodWithCacheTest, PropagatesLoadingErrorWithoutRemovingCache) {
  auto control = std::make_shared<FakeBatchControl>();
  const cache::InstallGuard guard(control);
  RecordingLoader loader;
  loader.result = Error::NotFound;
  EXPECT_EQ(
      load_method_with_cache(loader, "decode", loader.backend.c_str(), guard),
      Error::NotFound);
  EXPECT_EQ(loader.calls, 1);
  EXPECT_EQ(loader.loaded_method, "decode");
  EXPECT_EQ(loader.resolved, control);
  EXPECT_EQ(cache::CacheRegistry::global().get(loader.key), control);
}

TEST(LoadMethodWithCacheTest, InvalidBackendSkipsLoading) {
  auto control = std::make_shared<FakeBatchControl>();
  const cache::InstallGuard guard(control);
  RecordingLoader loader;
  const std::string too_long(256, 'x');
  for (const char* backend :
       std::array<const char*, 3>{nullptr, "", too_long.c_str()}) {
    EXPECT_EQ(
        load_method_with_cache(loader, "forward", backend, guard),
        Error::InvalidArgument);
  }
  EXPECT_EQ(loader.calls, 0);
  EXPECT_TRUE(loader.key.empty());
}

TEST(SampleFromRowTest, SelectsFloatRowWithoutMutatingNeighbors) {
  std::array<float, 9> data{
      9.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 9.0f};
  const auto original = data;
  const auto logits =
      executorch::extension::make_tensor_ptr({1, 3, 3}, data.data());
  executorch::extension::llm::Sampler sampler(3, 0.0f);
  EXPECT_EQ(sample_from_row(*logits, 0, sampler), Token{0});
  EXPECT_EQ(sample_from_row(*logits, 1, sampler), Token{1});
  EXPECT_EQ(sample_from_row(*logits, 2, sampler), Token{2});

  sampler.set_temperature(1.0f);
  ASSERT_TRUE(sample_from_row(*logits, 1, sampler));
  for (size_t i : {0, 1, 2, 6, 7, 8}) {
    EXPECT_EQ(data[i], original[i]);
  }
}

TEST(SampleFromRowTest, RejectsInvalidRowsScalarAndEmptyLogits) {
  const auto logits = executorch::extension::make_tensor_ptr(
      {2, 3}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});
  const auto scalar = executorch::extension::make_tensor_ptr(1.0f);
  const auto empty_vocab =
      executorch::extension::make_tensor_ptr({2, 0}, std::vector<float>{});
  const auto empty_rows =
      executorch::extension::make_tensor_ptr({0, 3}, std::vector<float>{});
  executorch::extension::llm::Sampler sampler(3, 0.0f);
  EXPECT_FALSE(sample_from_row(*logits, -1, sampler));
  EXPECT_FALSE(sample_from_row(*logits, 2, sampler));
  EXPECT_FALSE(sample_from_row(*scalar, 0, sampler));
  EXPECT_FALSE(sample_from_row(*empty_vocab, 0, sampler));
  EXPECT_FALSE(sample_from_row(*empty_rows, 0, sampler));
  EXPECT_EQ(sample_from_row(*logits, 1, sampler), Token{2});
}

} // namespace

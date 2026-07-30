/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cache_et.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/cache/sequence_cache.h>

#include <memory>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::extension::llm::cache::CacheBase;
using executorch::extension::llm::cache::CacheBuilderRegistry;
using executorch::extension::llm::cache::CacheConfig;
using executorch::extension::llm::cache::CacheRegistry;
using executorch::extension::llm::cache::CacheSession;
using executorch::extension::llm::cache::LayerConfig;
using executorch::extension::llm::cache::LayerPolicy;
using executorch::extension::llm::cache::make_unique_key;
using executorch::extension::llm::cache::SequenceCache;
using executorch::runtime::Error;
namespace et = executorch::extension::llm::cache::et;

namespace {
LayerConfig flat_layer() {
  return LayerConfig{LayerPolicy{LayerPolicy::Kind::Flat, 0}, 2, 8};
}
LayerConfig ring_layer(int window) {
  return LayerConfig{LayerPolicy{LayerPolicy::Kind::Ring, window}, 2, 8};
}
} // namespace

// Initializes the ExecuTorch PAL so the ET adapter's error paths (which ET_LOG)
// can run.
class CacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

// ---- Flat policy -----------------------------------------------------------

TEST_F(CacheTest, FlatPlanAppendsAndReadsAllHistory) {
  SequenceCache cache(CacheConfig{8, 1, {flat_layer()}});

  auto p0 = cache.plan(/*layer=*/0, /*position=*/0, /*T=*/4); // prefill
  ASSERT_TRUE(p0.has_value());
  EXPECT_EQ(p0->n_write, 1);
  EXPECT_EQ(p0->write[0].start, 0);
  EXPECT_EQ(p0->write[0].len, 4);
  EXPECT_EQ(p0->n_read, 1);
  EXPECT_EQ(p0->read[0].start, 0);
  EXPECT_EQ(p0->read[0].len, 4);
  EXPECT_EQ(p0->read_base_pos, 0);
  cache.commit(*p0); // accept the prefill step

  auto p1 = cache.plan(0, 4, 1); // decode
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->write[0].start, 4);
  EXPECT_EQ(p1->read[0].len, 5);
}

// ---- Ring policy: wrap, eviction, read base position -----------------------

TEST_F(CacheTest, RingPlanWrapsAndEvicts) {
  // window 4, max_write unset -> defaults to window -> ring of 2*4 - 1 = 7.
  SequenceCache cache(CacheConfig{100, 1, {ring_layer(4)}});

  // Prefill chunk of 4 at position 0: one write run, reads [0,4), no wrap.
  auto p0 = cache.plan(0, 0, 4);
  ASSERT_TRUE(p0.has_value());
  EXPECT_EQ(p0->n_write, 1);
  EXPECT_EQ(p0->write[0].start, 0);
  EXPECT_EQ(p0->write[0].len, 4);
  EXPECT_EQ(p0->n_read, 1);
  EXPECT_EQ(p0->read[0].start, 0);
  EXPECT_EQ(p0->read[0].len, 4);
  EXPECT_EQ(p0->read_base_pos, 0);
  cache.commit(*p0);

  // Decode at position 7: write wraps to slot 0 (7 % 7); the read window
  // [4,8) wraps the ring -> phys [4,6] then [0,0]. Positions 0-3 evicted.
  auto p1 = cache.plan(0, 7, 1);
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->n_write, 1);
  EXPECT_EQ(p1->write[0].start, 0);
  EXPECT_EQ(p1->write[0].len, 1);
  EXPECT_EQ(p1->n_read, 2);
  EXPECT_EQ(p1->read[0].start, 4);
  EXPECT_EQ(p1->read[0].len, 3);
  EXPECT_EQ(p1->read[1].start, 0);
  EXPECT_EQ(p1->read[1].len, 1);
  EXPECT_EQ(p1->read_base_pos, 4); // oldest retained is logical position 4

  // A step larger than max_write would overrun the ring -> rejected.
  EXPECT_FALSE(cache.plan(0, 0, 5).has_value());
}

// ---- Mixed flat/ring: one shared length across layers ----------------------

TEST_F(CacheTest, MixedFlatRingShareOneLength) {
  SequenceCache cache(CacheConfig{100, 2, {flat_layer(), ring_layer(4)}});

  // Same step drives both layers (T <= window); commit once.
  auto p = cache.plan(0, 0, 3);
  ASSERT_TRUE(p.has_value());
  ASSERT_TRUE(cache.plan(1, 0, 3).has_value());
  cache.commit(*p);

  // Length advanced once (not per layer): length == 3.
  EXPECT_TRUE(cache.can_extend(97));
  EXPECT_FALSE(cache.can_extend(98));
}

// ---- Admission / rewind ----------------------------------------------------

TEST_F(CacheTest, CanExtendBoundedByCapacity) {
  SequenceCache cache(CacheConfig{2, 1, {flat_layer()}});
  EXPECT_TRUE(cache.can_extend(2));
  auto p = cache.plan(0, 0, 2);
  ASSERT_TRUE(p.has_value());
  cache.commit(*p);
  EXPECT_FALSE(cache.can_extend(1));
  EXPECT_FALSE(cache.plan(0, 2, 1).has_value()); // exceeds capacity
}

TEST_F(CacheTest, FlatRewindsFreelyToZero) {
  SequenceCache cache(CacheConfig{8, 1, {flat_layer()}});
  auto p = cache.plan(0, 0, 5);
  ASSERT_TRUE(p.has_value());
  cache.commit(*p);
  EXPECT_TRUE(cache.rewind(0)); // flat retains all history
  EXPECT_FALSE(cache.rewind(1)); // cannot grow
  cache.clear();
  EXPECT_TRUE(cache.can_extend(8));
}

TEST_F(CacheTest, RewindBoundedByRingWindow) {
  SequenceCache cache(CacheConfig{100, 2, {flat_layer(), ring_layer(4)}});
  // Advance length to 10 in chunks of 2 (<= window).
  for (int pos = 0; pos < 10; pos += 2) {
    auto p = cache.plan(0, pos, 2);
    ASSERT_TRUE(p.has_value());
    ASSERT_TRUE(cache.plan(1, pos, 2).has_value());
    cache.commit(*p);
  }
  // ring(4) has evicted everything older than 10 - 4 = 6.
  EXPECT_FALSE(cache.rewind(5)); // older than the ring retains
  EXPECT_TRUE(cache.rewind(6)); // exactly the floor
  EXPECT_FALSE(cache.rewind(11)); // cannot grow
}

// ---- Faces / registry / session --------------------------------------------

TEST_F(CacheTest, FaceRecoveryReturnsSameObject) {
  SequenceCache cache(CacheConfig{4, 1, {flat_layer()}});
  CacheBase* base = &cache;
  ASSERT_NE(base->as_control(), nullptr);
  ASSERT_NE(base->as_planner(), nullptr);
  EXPECT_TRUE(base->as_control()->can_extend(4));
  auto plan = base->as_planner()->plan(0, 0, 1);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->read[0].len, 1);
}

TEST_F(CacheTest, RegistryInstallGetErase) {
  auto& reg = CacheRegistry::global();
  const std::string key = make_unique_key();
  EXPECT_EQ(reg.get(key), nullptr);

  std::shared_ptr<CacheBase> cache =
      std::make_shared<SequenceCache>(CacheConfig{16, 1, {flat_layer()}});
  reg.install(key, cache);
  EXPECT_EQ(reg.get(key), cache);
  EXPECT_TRUE(reg.get(key)->as_control()->can_extend(16));

  reg.erase(key);
  EXPECT_EQ(reg.get(key), nullptr);
}

TEST_F(CacheTest, UniqueKeysDoNotCollide) {
  EXPECT_NE(make_unique_key(), make_unique_key());
}

TEST_F(CacheTest, BuilderBuildsRegisteredKindElseError) {
  auto& reg = CacheBuilderRegistry::global();
  reg.register_builder("TestBackend", "seq", [](const CacheConfig& cfg) {
    return std::static_pointer_cast<CacheBase>(
        std::make_shared<SequenceCache>(cfg));
  });

  CacheConfig cfg{32, 1, {flat_layer()}};
  auto cache = reg.build("TestBackend", "seq", cfg);
  ASSERT_TRUE(cache.ok());
  EXPECT_EQ(cache.get()->as_control()->capacity(), 32);

  EXPECT_EQ(reg.build("TestBackend", "missing", cfg).error(), Error::NotFound);
}

TEST_F(CacheTest, SessionInstallsOnCtorErasesOnDtor) {
  const std::string key = make_unique_key();
  {
    CacheSession session(
        key,
        std::make_shared<SequenceCache>(CacheConfig{4, 1, {flat_layer()}}));
    EXPECT_NE(CacheRegistry::global().get(key), nullptr);
    EXPECT_TRUE(session.control()->can_extend(4));
  }
  EXPECT_EQ(CacheRegistry::global().get(key), nullptr);
}

// ---- ET adapter (maps core bool/optional to Error/Result) ------------------

TEST_F(CacheTest, EtAdapterMapsResultsAndCodes) {
  SequenceCache cache(CacheConfig{2, 1, {flat_layer()}});
  auto ok = et::plan(cache, /*layer=*/0, /*position=*/0, /*T=*/2);
  ASSERT_TRUE(ok.ok());
  EXPECT_EQ(ok->read[0].len, 2);
  cache.commit(ok.get()); // accept the step so rewind has history to truncate
  EXPECT_EQ(
      et::plan(cache, 0, 2, 1).error(), Error::OutOfResources); // over capacity
  EXPECT_FALSE(et::plan(cache, 5, 0, 1).ok()); // bad layer

  EXPECT_EQ(et::rewind(cache, 9), Error::InvalidArgument); // cannot grow
  EXPECT_EQ(et::rewind(cache, 1), Error::Ok);
}

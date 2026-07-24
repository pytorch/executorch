/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cache_registry.h>

#include <memory>

#include <gtest/gtest.h>

using executorch::extension::llm::cache::Cache;
using executorch::extension::llm::cache::CacheBuilderRegistry;
using executorch::extension::llm::cache::CacheConfig;
using executorch::extension::llm::cache::CacheRegistry;
using executorch::extension::llm::cache::CacheSession;
using executorch::extension::llm::cache::ContiguousBookkeeping;
using executorch::extension::llm::cache::make_unique_key;
using executorch::extension::llm::cache::StepPlan;

TEST(ContiguousBookkeepingTest, PlanAppendsAndAdvancesLength) {
  ContiguousBookkeeping bk(/*capacity=*/8);
  StepPlan p0 = bk.plan(/*position=*/0, /*T=*/4); // prefill
  EXPECT_TRUE(p0.contiguous);
  EXPECT_EQ(p0.write_start, 0);
  EXPECT_EQ(p0.write_len, 4);
  EXPECT_EQ(p0.valid_len, 4);

  StepPlan p1 = bk.plan(/*position=*/4, /*T=*/1); // decode
  EXPECT_EQ(p1.write_start, 4);
  EXPECT_EQ(p1.valid_len, 5);
}

TEST(ContiguousBookkeepingTest, CanExtendBoundsAtCapacity) {
  ContiguousBookkeeping bk(/*capacity=*/2);
  EXPECT_TRUE(bk.can_extend(2));
  bk.plan(/*position=*/0, /*T=*/2);
  EXPECT_FALSE(bk.can_extend(1));
  EXPECT_THROW(bk.plan(/*position=*/2, /*T=*/1), std::runtime_error);
}

TEST(ContiguousBookkeepingTest, RewindTruncatesAndClearsReset) {
  ContiguousBookkeeping bk(/*capacity=*/8);
  bk.plan(0, 5);
  bk.rewind(2);
  EXPECT_TRUE(bk.can_extend(6));
  EXPECT_THROW(bk.rewind(9), std::runtime_error);
  bk.clear();
  EXPECT_TRUE(bk.can_extend(8));
}

TEST(CacheRegistryTest, InstallGetErase) {
  auto& reg = CacheRegistry::global();
  const std::string key = make_unique_key();
  EXPECT_EQ(reg.get(key), nullptr);

  auto cache = std::make_shared<ContiguousBookkeeping>(16);
  reg.install(key, cache);
  EXPECT_EQ(reg.get(key), cache);

  reg.erase(key);
  EXPECT_EQ(reg.get(key), nullptr);
}

TEST(CacheRegistryTest, UniqueKeysDoNotCollide) {
  EXPECT_NE(make_unique_key(), make_unique_key());
}

TEST(CacheBuilderRegistryTest, BuildsRegisteredKindElseThrows) {
  auto& reg = CacheBuilderRegistry::global();
  reg.register_builder("TestBackend", "contiguous", [](const CacheConfig& cfg) {
    return std::static_pointer_cast<Cache>(
        std::make_shared<ContiguousBookkeeping>(cfg.capacity));
  });

  CacheConfig cfg{
      /*capacity=*/32,
      /*n_layers=*/1,
      /*n_kv_heads=*/{1},
      /*head_dim=*/{8}};
  auto cache = reg.build("TestBackend", "contiguous", cfg);
  ASSERT_NE(cache, nullptr);
  EXPECT_EQ(cache->capacity(), 32);

  EXPECT_THROW(reg.build("TestBackend", "missing", cfg), std::runtime_error);
}

TEST(CacheSessionTest, InstallsOnCtorErasesOnDtor) {
  const std::string key = make_unique_key();
  {
    CacheSession<Cache> session(
        key, std::make_shared<ContiguousBookkeeping>(4));
    EXPECT_NE(CacheRegistry::global().get(key), nullptr);
    EXPECT_TRUE(session->can_extend(4));
  }
  EXPECT_EQ(CacheRegistry::global().get(key), nullptr);
}

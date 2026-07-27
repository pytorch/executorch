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

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

using executorch::extension::llm::cache::CacheBase;
using executorch::extension::llm::cache::CacheBuilderRegistry;
using executorch::extension::llm::cache::CacheConfig;
using executorch::extension::llm::cache::CacheRegistry;
using executorch::extension::llm::cache::CacheSession;
using executorch::extension::llm::cache::make_unique_key;
using executorch::runtime::Error;

namespace {
// Opaque CacheBase used to exercise the registry without a concrete cache
// implementation (SequenceCache and its faces arrive in a later change).
class StubCache : public CacheBase {};
} // namespace

// Initializes the ExecuTorch PAL so registry error paths (which ET_LOG) can
// run.
class CacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executorch::runtime::runtime_init();
  }
};

TEST_F(CacheTest, RegistryInstallGetErase) {
  auto& reg = CacheRegistry::global();
  const std::string key = make_unique_key();
  EXPECT_EQ(reg.get(key), nullptr);

  std::shared_ptr<CacheBase> cache = std::make_shared<StubCache>();
  reg.install(key, cache);
  EXPECT_EQ(reg.get(key), cache);

  reg.erase(key);
  EXPECT_EQ(reg.get(key), nullptr);
}

TEST_F(CacheTest, UniqueKeysDoNotCollide) {
  EXPECT_NE(make_unique_key(), make_unique_key());
}

TEST_F(CacheTest, BuilderRegistryBuildsRegisteredKindElseError) {
  auto& reg = CacheBuilderRegistry::global();
  reg.register_builder("TestBackend", "stub", [](const CacheConfig&) {
    return std::static_pointer_cast<CacheBase>(std::make_shared<StubCache>());
  });

  CacheConfig cfg{32, 1};
  auto cache = reg.build("TestBackend", "stub", cfg);
  ASSERT_TRUE(cache.ok());
  EXPECT_NE(cache.get(), nullptr);

  EXPECT_EQ(reg.build("TestBackend", "missing", cfg).error(), Error::NotFound);
}

TEST_F(CacheTest, SessionInstallsOnCtorErasesOnDtor) {
  const std::string key = make_unique_key();
  {
    CacheSession session(key, std::make_shared<StubCache>());
    EXPECT_EQ(CacheRegistry::global().get(key), session.cache());
  }
  EXPECT_EQ(CacheRegistry::global().get(key), nullptr);
}

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/runtime/backend/interface.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace executorch::backends::cuda {
// Internal load-time resolver implemented by the CUDA delegate.
runtime::Result<std::string> resolve_cuda_cache_key(
    const runtime::BackendInitContext& context,
    runtime::ArrayRef<runtime::CompileSpec> compile_specs);
} // namespace executorch::backends::cuda

namespace {
namespace cache = executorch::extension::llm::cache;
namespace rt = executorch::runtime;
using executorch::backends::cuda::resolve_cuda_cache_key;

class MarkerCache final : public cache::Cache {
 protected:
  void* face(cache::FaceId) override {
    return nullptr;
  }
};

template <size_t N>
rt::BackendInitContext context(rt::BackendOptions<N>& options) {
  auto view = options.view();
  return rt::BackendInitContext(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      rt::Span<const rt::BackendOption>(view.data(), view.size()));
}

rt::CompileSpec role(const std::string& logical) {
  return {
      cache::kCacheKeyOption,
      {const_cast<char*>(logical.data()), logical.size()}};
}

std::string installed_key(const cache::InstallGuard& guard) {
  rt::BackendOptions<1> options;
  EXPECT_EQ(guard.set_option(options), rt::Error::Ok);
  const char* key = nullptr;
  EXPECT_EQ(options.get_option(cache::kCacheKeyOption, key), rt::Error::Ok);
  return key ? key : "";
}

TEST(CudaCacheRoutingTest, LegacyNoKeyKeepsInGraphBehavior) {
  rt::BackendOptions<1> options;
  auto resolved = resolve_cuda_cache_key(context(options), {});
  ASSERT_TRUE(resolved.ok());
  EXPECT_TRUE(resolved->empty());
}

TEST(CudaCacheRoutingTest, LegacyGuardOptionResolvesActualRegistryEntry) {
  auto cache_ptr = std::make_shared<MarkerCache>();
  cache::InstallGuard guard(cache_ptr);
  rt::BackendOptions<1> options;
  ASSERT_EQ(guard.set_option(options), rt::Error::Ok);
  auto resolved = resolve_cuda_cache_key(context(options), {});
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(cache::CacheRegistry::global().get(*resolved), cache_ptr);
}

TEST(CudaCacheRoutingTest, TargetAndDraftRolesMapToDistinctGuardKeys) {
  auto target = std::make_shared<MarkerCache>();
  auto draft = std::make_shared<MarkerCache>();
  cache::InstallGuard target_guard(target), draft_guard(draft);
  rt::BackendOptions<2> options;
  ASSERT_EQ(
      options.set_option(
          "muse_glimmer_target", installed_key(target_guard).c_str()),
      rt::Error::Ok);
  ASSERT_EQ(
      options.set_option(
          "muse_glimmer_draft", installed_key(draft_guard).c_str()),
      rt::Error::Ok);
  const std::string names[] = {"muse_glimmer_target", "muse_glimmer_draft"};
  const std::shared_ptr<MarkerCache> expected[] = {target, draft};
  for (size_t i = 0; i < 2; ++i) {
    auto spec = role(names[i]);
    auto resolved = resolve_cuda_cache_key(context(options), {&spec, 1});
    ASSERT_TRUE(resolved.ok());
    EXPECT_NE(*resolved, names[i]);
    EXPECT_EQ(cache::CacheRegistry::global().get(*resolved), expected[i]);
    EXPECT_EQ(cache::CacheRegistry::global().get(names[i]), nullptr);
  }
}

TEST(CudaCacheRoutingTest, PerMethodActualOverrideWinsOverLogicalMapping) {
  auto target = std::make_shared<MarkerCache>();
  auto draft = std::make_shared<MarkerCache>();
  cache::InstallGuard target_guard(target), draft_guard(draft);
  rt::BackendOptions<2> options;
  ASSERT_EQ(draft_guard.set_option(options), rt::Error::Ok);
  ASSERT_EQ(
      options.set_option(
          "muse_glimmer_draft", installed_key(target_guard).c_str()),
      rt::Error::Ok);
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  auto resolved = resolve_cuda_cache_key(context(options), {&spec, 1});
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(cache::CacheRegistry::global().get(*resolved), draft);
}

TEST(CudaCacheRoutingTest, PerMethodActualOverrideNeedsNoLogicalMapping) {
  auto draft = std::make_shared<MarkerCache>();
  cache::InstallGuard guard(draft);
  rt::BackendOptions<1> options;
  ASSERT_EQ(guard.set_option(options), rt::Error::Ok);
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  auto resolved = resolve_cuda_cache_key(context(options), {&spec, 1});
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(cache::CacheRegistry::global().get(*resolved), draft);
}

TEST(CudaCacheRoutingTest, SameLogicalRoleInTwoModelsDoesNotCollide) {
  auto a = std::make_shared<MarkerCache>(), b = std::make_shared<MarkerCache>();
  cache::InstallGuard ga(a), gb(b);
  rt::BackendOptions<1> oa, ob;
  ASSERT_EQ(
      oa.set_option("muse_glimmer_draft", installed_key(ga).c_str()),
      rt::Error::Ok);
  ASSERT_EQ(
      ob.set_option("muse_glimmer_draft", installed_key(gb).c_str()),
      rt::Error::Ok);
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  auto ra = resolve_cuda_cache_key(context(oa), {&spec, 1});
  auto rb = resolve_cuda_cache_key(context(ob), {&spec, 1});
  ASSERT_TRUE(ra.ok());
  ASSERT_TRUE(rb.ok());
  EXPECT_NE(*ra, *rb);
  EXPECT_EQ(cache::CacheRegistry::global().get(*ra), a);
  EXPECT_EQ(cache::CacheRegistry::global().get(*rb), b);
}

TEST(CudaCacheRoutingTest, MissingLogicalMappingIsNotARawRegistryLookup) {
  rt::BackendOptions<1> options;
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {&spec, 1}).error(),
      rt::Error::NotFound);
}

TEST(CudaCacheRoutingTest, EmptyGenericOverrideFallsBackToRoleMapping) {
  rt::BackendOptions<2> options;
  ASSERT_EQ(options.set_option(cache::kCacheKeyOption, ""), rt::Error::Ok);
  ASSERT_EQ(
      options.set_option("muse_glimmer_draft", "cache-test"), rt::Error::Ok);
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  auto resolved = resolve_cuda_cache_key(context(options), {&spec, 1});
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(*resolved, "cache-test");
}

TEST(CudaCacheRoutingTest, EmptyOrWrongTypeMappingIsRejected) {
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  rt::BackendOptions<1> options;
  ASSERT_EQ(options.set_option("muse_glimmer_draft", ""), rt::Error::Ok);
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {&spec, 1}).error(),
      rt::Error::InvalidArgument);
  ASSERT_EQ(options.set_option("muse_glimmer_draft", 42), rt::Error::Ok);
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {&spec, 1}).error(),
      rt::Error::InvalidArgument);
}

TEST(CudaCacheRoutingTest, WrongTypeOverrideIsRejectedForLogicalRole) {
  rt::BackendOptions<1> options;
  ASSERT_EQ(options.set_option(cache::kCacheKeyOption, true), rt::Error::Ok);
  const std::string logical = "muse_glimmer_draft";
  auto spec = role(logical);
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {&spec, 1}).error(),
      rt::Error::InvalidArgument);
}

TEST(CudaCacheRoutingTest, LegacyWrongTypeOverrideKeepsNoKeyBehavior) {
  rt::BackendOptions<1> options;
  ASSERT_EQ(options.set_option(cache::kCacheKeyOption, true), rt::Error::Ok);
  auto resolved = resolve_cuda_cache_key(context(options), {});
  ASSERT_TRUE(resolved.ok());
  EXPECT_TRUE(resolved->empty());
}

TEST(CudaCacheRoutingTest, MalformedCompileKeysAreRejectedEvenWithOverride) {
  rt::BackendOptions<1> options;
  ASSERT_EQ(
      options.set_option(cache::kCacheKeyOption, "cache-test"), rt::Error::Ok);
  const std::string invalid[] = {
      "",
      std::string("bad\0role", 8),
      std::string(rt::kMaxOptionKeyLength, 'x'),
      cache::kCacheKeyOption};
  for (const auto& logical : invalid) {
    auto spec = role(logical);
    EXPECT_EQ(
        resolve_cuda_cache_key(context(options), {&spec, 1}).error(),
        rt::Error::InvalidArgument);
  }
  rt::CompileSpec null_spec{cache::kCacheKeyOption, {nullptr, 4}};
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {&null_spec, 1}).error(),
      rt::Error::InvalidArgument);
}

TEST(CudaCacheRoutingTest, DuplicateCompileKeysAreRejected) {
  rt::BackendOptions<1> options;
  const std::string logical = "muse_glimmer_draft";
  rt::CompileSpec specs[] = {role(logical), role(logical)};
  EXPECT_EQ(
      resolve_cuda_cache_key(context(options), {specs, 2}).error(),
      rt::Error::InvalidArgument);
}

TEST(CudaCacheRoutingTest, OptionalTrailingNulIsAcceptedButNotEmbeddedNuls) {
  rt::BackendOptions<1> options;
  ASSERT_EQ(options.set_option("draft", "cache-test"), rt::Error::Ok);
  const std::string logical("draft\0", 6);
  auto spec = role(logical);
  auto resolved = resolve_cuda_cache_key(context(options), {&spec, 1});
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(*resolved, "cache-test");
}

TEST(
    CudaCacheRoutingTest,
    ResolvedKeyOwnsItsBytesAndAcquiredCacheOutlivesGuard) {
  auto cache_ptr = std::make_shared<MarkerCache>();
  std::string key;
  std::shared_ptr<cache::Cache> acquired;
  {
    cache::InstallGuard guard(cache_ptr);
    rt::BackendOptions<1> options;
    ASSERT_EQ(guard.set_option(options), rt::Error::Ok);
    auto resolved = resolve_cuda_cache_key(context(options), {});
    ASSERT_TRUE(resolved.ok());
    key = *resolved;
    ASSERT_EQ(
        options.set_option(cache::kCacheKeyOption, "changed"), rt::Error::Ok);
    EXPECT_EQ(*resolved, key);
    acquired = cache::CacheRegistry::global().get(*resolved);
    EXPECT_EQ(acquired, cache_ptr);
  }
  EXPECT_EQ(cache::CacheRegistry::global().get(key), nullptr);
  EXPECT_EQ(acquired, cache_ptr);
}
} // namespace

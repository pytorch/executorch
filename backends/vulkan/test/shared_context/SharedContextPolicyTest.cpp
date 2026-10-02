/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/SharedContext.h>
#include <executorch/runtime/backend/options.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace executorch::backends::vulkan_shared;
using executorch::backends::vulkan::parse_vulkan_shared_context_config;
using executorch::backends::vulkan::resolve_vulkan_shared_context;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendOption;
using executorch::runtime::BackendOptions;
using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::Span;

SharedVulkanContextPtr fake_context(const SharedVulkanContextKey& key) {
  SharedVulkanContextCreateInfo info;
  info.key = key;
  info.instance = reinterpret_cast<VkInstance>(uintptr_t{1});
  info.physical_device = reinterpret_cast<VkPhysicalDevice>(uintptr_t{2});
  info.device = reinterpret_cast<VkDevice>(uintptr_t{3});
  info.queue = reinterpret_cast<VkQueue>(uintptr_t{4});
  info.queue_family_index = 2;
  info.lifetime_anchor = std::make_shared<int>(7);
  return std::make_shared<SharedVulkanContext>(std::move(info));
}

template <size_t N>
BackendInitContext init_context(BackendOptions<N>& options) {
  auto view = options.view();
  return BackendInitContext(
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      Span<const BackendOption>(view.data(), view.size()));
}

class SharedContextPolicyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    registry().clear_for_testing();
  }
  void TearDown() override {
    registry().clear_for_testing();
  }
  SharedVulkanContextRegistry& registry() {
    return SharedVulkanContextRegistry::Get();
  }
  SharedVulkanRuntimeConfig config(SharedContextMode mode) {
    SharedVulkanRuntimeConfig result;
    result.context_name = "test-policy";
    result.group_id = 7;
    result.context_mode = mode;
    return result;
  }
  SharedVulkanContextKey key() {
    return {"test-policy", 7};
  }
};

TEST_F(SharedContextPolicyTest, NoOptionsPreserveLegacy) {
  BackendInitContext context(nullptr);
  auto parsed = parse_vulkan_shared_context_config(context);
  ASSERT_TRUE(parsed.ok());
  EXPECT_FALSE(parsed->enabled());
  auto base = parse_shared_vulkan_runtime_config(context);
  ASSERT_TRUE(base.ok());
  EXPECT_TRUE(base->lookup_or_create());
}

TEST_F(SharedContextPolicyTest, UnrelatedOptionsPreserveLegacy) {
  BackendOptions<1> options;
  ASSERT_EQ(options.set_option("unrelated_option", 12), Error::Ok);
  auto parsed = parse_vulkan_shared_context_config(init_context(options));
  ASSERT_TRUE(parsed.ok());
  EXPECT_FALSE(parsed->enabled());
}

TEST_F(SharedContextPolicyTest, EachSharedOptionOptsIn) {
  BackendOptions<1> context_name;
  ASSERT_EQ(
      context_name.set_option(kSharedContextNameOption, "custom"), Error::Ok);
  auto by_token =
      parse_vulkan_shared_context_config(init_context(context_name));
  ASSERT_TRUE(by_token.ok());
  EXPECT_TRUE(by_token->lookup_or_create());
  EXPECT_EQ(by_token->context_name, "custom");
  BackendOptions<1> group;
  ASSERT_EQ(group.set_option(kSharedGroupIdOption, -5), Error::Ok);
  auto by_group = parse_vulkan_shared_context_config(init_context(group));
  ASSERT_TRUE(by_group.ok());
  EXPECT_TRUE(by_group->enabled());
  EXPECT_EQ(by_group->group_id, -5);
  BackendOptions<1> mode;
  ASSERT_EQ(
      mode.set_option(kSharedContextModeOption, "lookup_only"), Error::Ok);
  auto by_mode = parse_vulkan_shared_context_config(init_context(mode));
  ASSERT_TRUE(by_mode.ok());
  EXPECT_TRUE(by_mode->lookup_only());
}

TEST_F(SharedContextPolicyTest, ExplicitDisabledAllowsEmptyContextName) {
  BackendOptions<2> options;
  ASSERT_EQ(options.set_option(kSharedContextNameOption, ""), Error::Ok);
  ASSERT_EQ(
      options.set_option(kSharedContextModeOption, "disabled"), Error::Ok);
  auto parsed = parse_vulkan_shared_context_config(init_context(options));
  ASSERT_TRUE(parsed.ok());
  EXPECT_FALSE(parsed->enabled());
}

TEST_F(SharedContextPolicyTest, RejectsBadRuntimeOptionTypesAndMode) {
  BackendOptions<1> context_name;
  ASSERT_EQ(context_name.set_option(kSharedContextNameOption, 1), Error::Ok);
  EXPECT_EQ(
      parse_vulkan_shared_context_config(init_context(context_name)).error(),
      Error::InvalidArgument);
  BackendOptions<1> group;
  ASSERT_EQ(group.set_option(kSharedGroupIdOption, "7"), Error::Ok);
  EXPECT_EQ(
      parse_vulkan_shared_context_config(init_context(group)).error(),
      Error::InvalidArgument);
  BackendOptions<1> mode;
  ASSERT_EQ(mode.set_option(kSharedContextModeOption, true), Error::Ok);
  EXPECT_EQ(
      parse_vulkan_shared_context_config(init_context(mode)).error(),
      Error::InvalidArgument);
  ASSERT_EQ(mode.set_option(kSharedContextModeOption, "invalid"), Error::Ok);
  EXPECT_EQ(
      parse_vulkan_shared_context_config(init_context(mode)).error(),
      Error::InvalidArgument);
  BackendOptions<1> empty;
  ASSERT_EQ(empty.set_option(kSharedContextNameOption, ""), Error::Ok);
  EXPECT_EQ(
      parse_vulkan_shared_context_config(init_context(empty)).error(),
      Error::InvalidArgument);
}

TEST_F(SharedContextPolicyTest, DisabledDoesNotLookupOrCreate) {
  auto existing = fake_context(key());
  ASSERT_EQ(registry().register_context(existing), Error::Ok);
  int calls = 0;
  auto result = resolve_vulkan_shared_context(
      config(SharedContextMode::kDisabled), [&]() {
        ++calls;
        return Result<SharedVulkanContextPtr>(existing);
      });
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), nullptr);
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(registry().lookup(key()), existing);
}

TEST_F(SharedContextPolicyTest, LookupOnlyMissingNeverCreates) {
  int calls = 0;
  auto result = resolve_vulkan_shared_context(
      config(SharedContextMode::kLookupOnly), [&]() {
        ++calls;
        return Result<SharedVulkanContextPtr>(fake_context(key()));
      });
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::NotFound);
  EXPECT_EQ(calls, 0);
}

TEST_F(SharedContextPolicyTest, LookupOnlyReusesRegisteredContext) {
  auto existing = fake_context(key());
  ASSERT_EQ(registry().register_context(existing), Error::Ok);
  auto result =
      resolve_vulkan_shared_context(config(SharedContextMode::kLookupOnly), {});
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get(), existing);
}

TEST_F(SharedContextPolicyTest, LookupOrCreateCreatesExactlyOnce) {
  int calls = 0;
  auto creator = [&]() -> Result<SharedVulkanContextPtr> {
    ++calls;
    return fake_context(key());
  };
  auto first = resolve_vulkan_shared_context(
      config(SharedContextMode::kLookupOrCreate), creator);
  auto second = resolve_vulkan_shared_context(
      config(SharedContextMode::kLookupOrCreate), creator);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.get(), second.get());
  EXPECT_EQ(calls, 1);
}

TEST_F(SharedContextPolicyTest, CreateOnlyRejectsDuplicateWithoutCreating) {
  auto c = config(SharedContextMode::kCreateOnly);
  int calls = 0;
  auto creator = [&]() -> Result<SharedVulkanContextPtr> {
    ++calls;
    return fake_context(key());
  };
  ASSERT_TRUE(resolve_vulkan_shared_context(c, creator).ok());
  auto duplicate = resolve_vulkan_shared_context(c, creator);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error(), Error::AlreadyLoaded);
  EXPECT_EQ(calls, 1);
}

TEST_F(SharedContextPolicyTest, ContextNameAndGroupAreBothIdentity) {
  auto c = config(SharedContextMode::kLookupOrCreate);
  auto first =
      resolve_vulkan_shared_context(c, [&]() -> Result<SharedVulkanContextPtr> {
        return fake_context({c.context_name, c.group_id});
      });
  ASSERT_TRUE(first.ok());
  for (const auto& other : std::vector<SharedVulkanContextKey>{
           {c.context_name, -7}, {"other", 7}}) {
    c.context_name = other.context_name;
    c.group_id = other.group_id;
    auto next = resolve_vulkan_shared_context(
        c, [&]() -> Result<SharedVulkanContextPtr> {
          return fake_context(other);
        });
    ASSERT_TRUE(next.ok());
    EXPECT_NE(first.get(), next.get());
  }
}

TEST_F(SharedContextPolicyTest, RejectsInvalidDirectConfigAndMissingCreator) {
  auto c = config(static_cast<SharedContextMode>(255));
  EXPECT_EQ(
      resolve_vulkan_shared_context(
          c, []() -> Result<SharedVulkanContextPtr> { return Error::Internal; })
          .error(),
      Error::InvalidArgument);
  c = config(SharedContextMode::kCreateOnly);
  EXPECT_EQ(
      resolve_vulkan_shared_context(c, {}).error(), Error::InvalidArgument);
  c.context_name.clear();
  EXPECT_EQ(
      resolve_vulkan_shared_context(c, {}).error(), Error::InvalidArgument);
}

TEST_F(SharedContextPolicyTest, CreatorErrorDoesNotPoisonRetry) {
  auto c = config(SharedContextMode::kLookupOrCreate);
  auto failure =
      resolve_vulkan_shared_context(c, []() -> Result<SharedVulkanContextPtr> {
        return Error::DelegateInvalidCompatibility;
      });
  ASSERT_FALSE(failure.ok());
  EXPECT_EQ(failure.error(), Error::DelegateInvalidCompatibility);
  EXPECT_EQ(registry().lookup(key()), nullptr);
  EXPECT_TRUE(
      resolve_vulkan_shared_context(c, [&]() -> Result<SharedVulkanContextPtr> {
        return fake_context(key());
      }).ok());
}

TEST_F(SharedContextPolicyTest, CreatorExceptionsDoNotLeaveCreationInFlight) {
  auto c = config(SharedContextMode::kLookupOrCreate);
  auto bad_alloc = resolve_vulkan_shared_context(
      c, []() -> Result<SharedVulkanContextPtr> { throw std::bad_alloc(); });
  EXPECT_EQ(bad_alloc.error(), Error::MemoryAllocationFailed);
  auto exception =
      resolve_vulkan_shared_context(c, []() -> Result<SharedVulkanContextPtr> {
        throw std::runtime_error("injected creation failure");
      });
  EXPECT_EQ(exception.error(), Error::Internal);
  EXPECT_TRUE(
      resolve_vulkan_shared_context(c, [&]() -> Result<SharedVulkanContextPtr> {
        return fake_context(key());
      }).ok());
}

TEST_F(SharedContextPolicyTest, NullInvalidAndWrongKeyContextsAreRejected) {
  auto c = config(SharedContextMode::kLookupOrCreate);
  for (auto candidate : std::vector<SharedVulkanContextPtr>{
           nullptr,
           std::make_shared<SharedVulkanContext>(
               SharedVulkanContextCreateInfo{}),
           fake_context({"wrong", 7})}) {
    auto result = resolve_vulkan_shared_context(
        c,
        [candidate]() -> Result<SharedVulkanContextPtr> { return candidate; });
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.error(), Error::InvalidArgument);
    EXPECT_EQ(registry().lookup(key()), nullptr);
  }
}

TEST_F(SharedContextPolicyTest, ConcurrentLookupOrCreateUsesOneCreator) {
  const auto c = config(SharedContextMode::kLookupOrCreate);
  std::atomic<int> calls{0};
  std::array<SharedVulkanContextPtr, 8> contexts;
  std::array<Error, 8> errors{};
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::vector<std::thread> threads;
  for (size_t i = 0; i < contexts.size(); ++i) {
    threads.emplace_back([&, i] {
      gate.wait();
      auto result = resolve_vulkan_shared_context(
          c, [&]() -> Result<SharedVulkanContextPtr> {
            ++calls;
            return fake_context({c.context_name, c.group_id});
          });
      errors[i] = result.ok() ? Error::Ok : result.error();
      if (result.ok()) {
        contexts[i] = result.get();
      }
    });
  }
  start.set_value();
  for (auto& thread : threads) {
    thread.join();
  }
  for (size_t i = 0; i < contexts.size(); ++i) {
    EXPECT_EQ(errors[i], Error::Ok);
    EXPECT_EQ(contexts[i], contexts[0]);
  }
  EXPECT_NE(contexts[0], nullptr);
  EXPECT_EQ(calls, 1);
}

TEST_F(SharedContextPolicyTest, ConcurrentCreateOnlyHasOneWinner) {
  auto c = config(SharedContextMode::kCreateOnly);
  std::atomic<int> calls{0}, winners{0}, duplicates{0}, unexpected{0};
  std::promise<void> start;
  auto gate = start.get_future().share();
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&] {
      gate.wait();
      auto result = resolve_vulkan_shared_context(
          c, [&]() -> Result<SharedVulkanContextPtr> {
            ++calls;
            return fake_context({c.context_name, c.group_id});
          });
      if (result.ok()) {
        ++winners;
      } else if (result.error() == Error::AlreadyLoaded) {
        ++duplicates;
      } else {
        ++unexpected;
      }
    });
  }
  start.set_value();
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(winners, 1);
  EXPECT_EQ(duplicates, 7);
  EXPECT_EQ(unexpected, 0);
}

TEST_F(SharedContextPolicyTest, ExternalRegistrationCanWinAgainstCreateOnly) {
  const auto c = config(SharedContextMode::kCreateOnly);
  std::promise<void> entered, release;
  auto gate = release.get_future();
  auto future = std::async(std::launch::async, [&] {
    return resolve_vulkan_shared_context(
        c, [&]() -> Result<SharedVulkanContextPtr> {
          entered.set_value();
          gate.wait();
          return fake_context({c.context_name, c.group_id});
        });
  });
  entered.get_future().wait();
  auto external = fake_context(key());
  const auto registered = registry().register_context(external);
  release.set_value(); // Always unblock before assertions can return.
  auto result = future.get();
  EXPECT_EQ(registered, Error::Ok);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::AlreadyLoaded);
  EXPECT_EQ(registry().lookup(key()), external);
}

} // namespace

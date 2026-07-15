/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/gpu_shared/runtime/SharedGpuContextRegistry.h>

#include <gtest/gtest.h>
// Cppcheck's lint environment may not expand the gtest macros.
#ifndef TEST
#define TEST(test_suite_name, test_name) void test_suite_name##_##test_name()
#endif
#ifndef TEST_F
#define TEST_F(test_fixture, test_name) void test_fixture##_##test_name()
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using executorch::backends::gpu_shared::SharedGpuContext;
using executorch::backends::gpu_shared::SharedGpuContextCreateInfo;
using executorch::backends::gpu_shared::SharedGpuContextKey;
using executorch::backends::gpu_shared::SharedGpuContextPtr;
using executorch::backends::gpu_shared::SharedGpuContextRegistry;
using executorch::runtime::Error;
using executorch::runtime::Result;

namespace {

template <typename Handle>
Handle fake_handle(uintptr_t value) {
  return reinterpret_cast<Handle>(value);
}

SharedGpuContextCreateInfo make_create_info(
    SharedGpuContextKey key,
    uintptr_t handle_base = 1) {
  SharedGpuContextCreateInfo info;
  info.key = std::move(key);
  info.instance = fake_handle<VkInstance>(handle_base);
  info.physical_device = fake_handle<VkPhysicalDevice>(handle_base + 1);
  info.device = fake_handle<VkDevice>(handle_base + 2);
  info.queue = fake_handle<VkQueue>(handle_base + 3);
  info.queue_family_index = 4;
  // Tests use a dummy owner by default now that every valid context requires a
  // lifetime anchor. Tests that exercise ownership replace this anchor.
  info.lifetime_anchor = std::make_shared<int>(0);
  return info;
}

struct RegistryLookupProbeState final {
  std::mutex mutex;
  std::condition_variable cv;
  bool complete = false;
};

class RegistryLookupLifetimeAnchor final {
 public:
  RegistryLookupLifetimeAnchor(
      SharedGpuContextKey lookup_key,
      std::shared_ptr<RegistryLookupProbeState> state,
      std::atomic<bool>* lookup_completed_during_destruction)
      : lookup_key_(std::move(lookup_key)),
        state_(std::move(state)),
        lookup_completed_during_destruction_(
            lookup_completed_during_destruction) {}

  ~RegistryLookupLifetimeAnchor() {
    // Probe the registry from another thread. If this destructor runs while the
    // registry mutex is held, lookup() cannot complete until destruction
    // returns and unregister_context() releases the mutex. The timeout prevents
    // the regression test itself from deadlocking on the buggy implementation.
    std::thread([lookup_key = lookup_key_, state = state_]() {
      (void)SharedGpuContextRegistry::Get().lookup(lookup_key);
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->complete = true;
      }
      state->cv.notify_all();
    }).detach();

    std::unique_lock<std::mutex> lock(state_->mutex);
    const bool completed =
        state_->cv.wait_for(lock, std::chrono::seconds(1), [state = state_]() {
          return state->complete;
        });
    lookup_completed_during_destruction_->store(completed);
  }

 private:
  SharedGpuContextKey lookup_key_;
  std::shared_ptr<RegistryLookupProbeState> state_;
  std::atomic<bool>* lookup_completed_during_destruction_;
};

class SharedGpuContextRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SharedGpuContextRegistry::Get().clear_for_testing();
  }

  void TearDown() override {
    SharedGpuContextRegistry::Get().clear_for_testing();
  }
};

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, RegistryOwnsPersistentContext) {
  const SharedGpuContextKey key{"scene0", 3};
  std::weak_ptr<int> owner_weak;

  {
    auto owner = std::make_shared<int>(17);
    owner_weak = owner;
    auto info = make_create_info(key);
    info.lifetime_anchor = owner;

    auto registered = SharedGpuContextRegistry::Get().register_external_context(
        std::move(info));
    ASSERT_TRUE(registered.ok());
  }

  EXPECT_FALSE(owner_weak.expired());
  EXPECT_NE(SharedGpuContextRegistry::Get().lookup(key), nullptr);
  EXPECT_EQ(SharedGpuContextRegistry::Get().unregister_context(key), Error::Ok);
  EXPECT_TRUE(owner_weak.expired());
}

// cppcheck-suppress unusedFunction
TEST_F(
    SharedGpuContextRegistryTest,
    UnregisterDestroysLifetimeAnchorOutsideRegistryLock) {
  const SharedGpuContextKey key{"scene0", 9};
  const SharedGpuContextKey probe_key{"probe", 9};
  std::atomic<bool> lookup_completed_during_destruction{false};
  auto probe_state = std::make_shared<RegistryLookupProbeState>();

  {
    auto info = make_create_info(key);
    info.lifetime_anchor = std::make_shared<RegistryLookupLifetimeAnchor>(
        probe_key, probe_state, &lookup_completed_during_destruction);

    auto registered = SharedGpuContextRegistry::Get().register_external_context(
        std::move(info));
    ASSERT_TRUE(registered.ok());
  }

  // The registry is now the only owner of the SharedGpuContext. Unregistering
  // therefore destroys its lifetime anchor. The probe must be able to acquire
  // the registry mutex before that destruction returns.
  EXPECT_EQ(SharedGpuContextRegistry::Get().unregister_context(key), Error::Ok);
  EXPECT_TRUE(lookup_completed_during_destruction.load());

  // On a broken implementation the probe only completes after unregister has
  // released the mutex. Wait for it here so the detached thread cannot escape
  // the test and race fixture teardown.
  std::unique_lock<std::mutex> lock(probe_state->mutex);
  EXPECT_TRUE(
      probe_state->cv.wait_for(lock, std::chrono::seconds(1), [probe_state]() {
        return probe_state->complete;
      }));
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, LookupOrCreateRunsCreatorOnce) {
  const SharedGpuContextKey key{"scene0", 4};
  std::atomic<int> create_count{0};
  std::mutex creator_mutex;
  std::condition_variable creator_cv;
  bool creator_entered = false;
  bool allow_creator_to_finish = false;

  auto create_fn = [&]() -> Result<SharedGpuContextPtr> {
    ++create_count;
    {
      std::unique_lock<std::mutex> lock(creator_mutex);
      creator_entered = true;
      creator_cv.notify_all();
      creator_cv.wait(lock, [&]() { return allow_creator_to_finish; });
    }
    return std::make_shared<SharedGpuContext>(make_create_info(key));
  };

  constexpr size_t kThreadCount = 8;
  std::vector<SharedGpuContextPtr> results(kThreadCount);
  std::vector<Error> errors(kThreadCount, Error::Internal);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i]() {
      auto result =
          SharedGpuContextRegistry::Get().lookup_or_create(key, create_fn);
      errors[i] = result.error();
      if (result.ok()) {
        results[i] = result.get();
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(creator_mutex);
    creator_cv.wait(lock, [&]() { return creator_entered; });
    allow_creator_to_finish = true;
  }
  creator_cv.notify_all();

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(create_count.load(), 1);
  for (size_t i = 0; i < kThreadCount; ++i) {
    EXPECT_EQ(errors[i], Error::Ok);
    EXPECT_EQ(results[i], results[0]);
  }
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, RejectsDifferentDuplicateContext) {
  const SharedGpuContextKey key{"scene0", 5};
  auto first = std::make_shared<SharedGpuContext>(make_create_info(key, 10));
  auto second = std::make_shared<SharedGpuContext>(make_create_info(key, 20));

  EXPECT_EQ(SharedGpuContextRegistry::Get().register_context(first), Error::Ok);
  EXPECT_EQ(
      SharedGpuContextRegistry::Get().register_context(second),
      Error::AlreadyLoaded);
  EXPECT_EQ(SharedGpuContextRegistry::Get().lookup(key), first);
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, ValidatesContextIdentity) {
  const SharedGpuContextKey requested_key{"scene0", 6};
  const SharedGpuContextKey returned_key{"other", 6};

  auto result = SharedGpuContextRegistry::Get().lookup_or_create(
      requested_key, [&]() -> Result<SharedGpuContextPtr> {
        return std::make_shared<SharedGpuContext>(
            make_create_info(returned_key));
      });

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
  EXPECT_EQ(SharedGpuContextRegistry::Get().lookup(requested_key), nullptr);
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, ReportsDeclaredDeviceExtensions) {
  const SharedGpuContextKey key{"scene0", 7};
  auto info = make_create_info(key);
  info.enabled_device_extensions = {"VK_ARM_tensors", "VK_ARM_data_graph"};
  SharedGpuContext context(std::move(info));

  EXPECT_TRUE(context.has_device_extension("VK_ARM_tensors"));
  EXPECT_TRUE(context.has_device_extension("VK_ARM_data_graph"));
  EXPECT_FALSE(context.has_device_extension("VK_KHR_nonexistent"));
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, RejectsContextWithoutLifetimeAnchor) {
  const SharedGpuContextKey key{"scene0", 10};
  auto info = make_create_info(key);
  info.lifetime_anchor.reset();

  auto result = SharedGpuContextRegistry::Get().register_external_context(
      std::move(info));

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

// cppcheck-suppress unusedFunction
TEST_F(
    SharedGpuContextRegistryTest,
    UnregisterKeepsLifetimeAnchorAliveWhileContextIsReferenced) {
  const SharedGpuContextKey key{"scene0", 11};
  std::weak_ptr<int> owner_weak;
  SharedGpuContextPtr held_context;

  {
    auto owner = std::make_shared<int>(17);
    owner_weak = owner;

    auto info = make_create_info(key);
    info.lifetime_anchor = owner;

    auto registered = SharedGpuContextRegistry::Get().register_external_context(
        std::move(info));
    ASSERT_TRUE(registered.ok());

    held_context = SharedGpuContextRegistry::Get().lookup(key);
    ASSERT_NE(held_context, nullptr);
  }

  EXPECT_FALSE(owner_weak.expired());
  EXPECT_EQ(SharedGpuContextRegistry::Get().unregister_context(key), Error::Ok);
  EXPECT_EQ(SharedGpuContextRegistry::Get().lookup(key), nullptr);

  // unregister_context() removes registry ownership only. An existing delegate
  // reference must continue to keep the underlying Vulkan objects alive.
  EXPECT_FALSE(owner_weak.expired());

  held_context.reset();
  EXPECT_TRUE(owner_weak.expired());
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, SerializesSharedQueueAccess) {
  const SharedGpuContextKey key{"scene0", 12};
  auto context = std::make_shared<SharedGpuContext>(make_create_info(key));

  std::mutex state_mutex;
  std::condition_variable state_cv;
  bool first_entered = false;
  bool release_first = false;
  bool second_started = false;
  bool second_entered = false;

  std::thread first([&]() {
    context->with_locked_queue([&](VkQueue) {
      std::unique_lock<std::mutex> lock(state_mutex);
      first_entered = true;
      state_cv.notify_all();
      state_cv.wait(lock, [&]() { return release_first; });
    });
  });

  bool first_ready = false;
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    first_ready = state_cv.wait_for(
        lock, std::chrono::seconds(1), [&]() { return first_entered; });
  }
  EXPECT_TRUE(first_ready);
  if (!first_ready) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      release_first = true;
    }
    state_cv.notify_all();
    first.join();
    return;
  }

  std::thread second([&]() {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      second_started = true;
    }
    state_cv.notify_all();

    context->with_locked_queue([&](VkQueue) {
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        second_entered = true;
      }
      state_cv.notify_all();
    });
  });

  bool second_ready = false;
  bool entered_while_first_held_queue_lock = false;
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    second_ready = state_cv.wait_for(
        lock, std::chrono::seconds(1), [&]() { return second_started; });
    if (second_ready) {
      entered_while_first_held_queue_lock =
          state_cv.wait_for(lock, std::chrono::milliseconds(100), [&]() {
            return second_entered;
          });
    }

    release_first = true;
  }
  state_cv.notify_all();

  EXPECT_TRUE(second_ready);
  EXPECT_FALSE(entered_while_first_held_queue_lock);

  bool second_completed = false;
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    second_completed = state_cv.wait_for(
        lock, std::chrono::seconds(1), [&]() { return second_entered; });
  }
  EXPECT_TRUE(second_completed);

  first.join();
  second.join();
}

// cppcheck-suppress unusedFunction
TEST_F(SharedGpuContextRegistryTest, RejectsIncompleteExternalContext) {
  SharedGpuContextCreateInfo info;
  info.key = {"scene0", 8};

  auto result = SharedGpuContextRegistry::Get().register_external_context(
      std::move(info));

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
}

} // namespace

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace executorch::backends::qnn {

using runtime::Error;

class QnnExecuTorchBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    runtime::runtime_init();
  }

  using DelegateKey = QnnExecuTorchBackend::DelegateKey;
  using SpillFillKey = QnnExecuTorchBackend::SpillFillKey;
  using SpillFillGroup = QnnExecuTorchBackend::SpillFillGroup;

  std::optional<void*> acquire(const DelegateKey& key) {
    return backend_.get_cached_delegate(key);
  }

  void* publish(
      const DelegateKey& key,
      void* handle,
      std::shared_ptr<void> manager = nullptr,
      std::shared_ptr<SpillFillGroup> group = nullptr) {
    return backend_.add_cached_delegate(key, handle, manager, group);
  }

  std::shared_ptr<SpillFillGroup> spill_fill_group(const SpillFillKey& key) {
    return backend_.get_spill_fill_group(key);
  }

  bool contains(void* handle) {
    return backend_.is_cached_delegate(handle);
  }

  bool release(void* handle) {
    return backend_.erase_cached_delegate(handle).has_value();
  }

  void leave_registered_delegate_alive() {
    auto* registered = static_cast<QnnExecuTorchBackend*>(
        runtime::get_backend_class("QnnBackend"));
    std::shared_ptr<void> manager(new int, [](void* handle) {
      delete static_cast<int*>(handle);
      std::_Exit(42);
    });
    registered->add_cached_delegate({42, nullptr}, manager.get(), manager);
  }

  QnnExecuTorchBackend backend_;
  decltype(QnnExecuTorchBackend::delegate_map_rev_)& reverse_map_ =
      backend_.delegate_map_rev_;
};

TEST_F(QnnExecuTorchBackendTest, AcceptsLpaiPriorities) {
  runtime::BackendOptionContext context;
  runtime::BackendOptions<1> options;
  for (int priority : {0, 50, 100, 150, 200, 300, 400, 500}) {
    SCOPED_TRACE(priority);
    ASSERT_EQ(
        options.set_option(QNN_RUNTIME_LPAI_PRIORITY, priority), Error::Ok);
    auto view = options.view();
    ASSERT_EQ(backend_.set_option(context, view), Error::Ok);
    view[0].value = -1;
    ASSERT_EQ(backend_.get_option(context, view), Error::Ok);
    EXPECT_EQ(std::get<int>(view[0].value), priority);
  }
}

TEST_F(
    QnnExecuTorchBackendTest,
    RejectsInvalidLpaiPrioritiesWithoutChangingState) {
  runtime::BackendOptionContext context;
  runtime::BackendOptions<1> options;
  ASSERT_EQ(options.set_option(QNN_RUNTIME_LPAI_PRIORITY, 100), Error::Ok);
  auto view = options.view();
  EXPECT_EQ(backend_.get_option(context, view), Error::Internal);
  const std::array<runtime::OptionValue, 6> invalid_values{
      -1,
      125,
      501,
      0x7fffffff,
      true,
      std::array<char, runtime::kMaxOptionValueLength>{},
  };
  for (bool already_set : {false, true}) {
    if (already_set) {
      view[0].value = 100;
      ASSERT_EQ(backend_.set_option(context, view), Error::Ok);
    }
    for (const auto& invalid : invalid_values) {
      view[0].value = invalid;
      EXPECT_EQ(backend_.set_option(context, view), Error::InvalidArgument);
      EXPECT_EQ(
          backend_.get_option(context, view),
          already_set ? Error::Ok : Error::Internal);
      if (already_set) {
        EXPECT_EQ(std::get<int>(view[0].value), 100);
      }
    }
  }
}

TEST_F(
    QnnExecuTorchBackendTest,
    InvalidLpaiPriorityLeavesBatchOptionsUnchanged) {
  runtime::BackendOptionContext context;
  runtime::BackendOptions<3> initial;
  ASSERT_EQ(initial.set_option(QNN_RUNTIME_LPAI_FPS, 30), Error::Ok);
  ASSERT_EQ(initial.set_option(QNN_RUNTIME_LPAI_PRIORITY, 100), Error::Ok);
  ASSERT_EQ(
      initial.set_option(QNN_RUNTIME_HEAP_PROFILING_PATH, "initial"),
      Error::Ok);
  auto initial_view = initial.view();
  const std::array<runtime::OptionValue, 2> invalid_priorities{125, true};
  for (bool already_set : {false, true}) {
    SCOPED_TRACE(already_set);
    if (already_set) {
      ASSERT_EQ(backend_.set_option(context, initial_view), Error::Ok);
    }
    for (const auto& invalid_priority : invalid_priorities) {
      runtime::BackendOptions<3> update = initial;
      ASSERT_EQ(update.set_option(QNN_RUNTIME_LPAI_FPS, 60), Error::Ok);
      ASSERT_EQ(
          update.set_option(QNN_RUNTIME_HEAP_PROFILING_PATH, "updated"),
          Error::Ok);
      auto update_view = update.view();
      update_view[1].value = invalid_priority;
      EXPECT_EQ(
          backend_.set_option(context, update_view), Error::InvalidArgument);
      for (size_t i = 0; i < update_view.size(); ++i) {
        runtime::Span<runtime::BackendOption> option(&update_view[i], 1);
        EXPECT_EQ(
            backend_.get_option(context, option),
            already_set ? Error::Ok : Error::Internal);
        if (already_set) {
          EXPECT_EQ(option[0].value, initial_view[i].value);
        }
      }
    }
  }
}

TEST_F(
    QnnExecuTorchBackendTest,
    RegisteredBackendSurvivesProcessExitWithLiveDelegate) {
  ASSERT_NE(runtime::get_backend_class("QnnBackend"), nullptr);
  EXPECT_EXIT(
      {
        leave_registered_delegate_alive();
        std::exit(0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST_F(
    QnnExecuTorchBackendTest,
    RemovesStaleReverseEntryWithoutDereferencingEnd) {
  int storage;
  DelegateKey key{42, nullptr};
  ASSERT_TRUE(reverse_map_.emplace(&storage, key).second);
  EXPECT_TRUE(contains(&storage));
  EXPECT_FALSE(release(&storage));
  EXPECT_FALSE(contains(&storage));
  EXPECT_EQ(acquire(key), std::nullopt);
  EXPECT_FALSE(release(&storage));
}

TEST_F(QnnExecuTorchBackendTest, SharesUntilLastMethodIsReleased) {
  int storage;
  DelegateKey key{42, nullptr};
  ASSERT_EQ(acquire(key), std::nullopt);
  ASSERT_EQ(publish(key, &storage), &storage);
  ASSERT_EQ(acquire(key), &storage);
  EXPECT_FALSE(release(&storage));
  EXPECT_TRUE(contains(&storage));
  EXPECT_TRUE(release(&storage));
  EXPECT_FALSE(contains(&storage));
  EXPECT_EQ(acquire(key), std::nullopt);
  EXPECT_FALSE(release(&storage));
  EXPECT_EQ(publish(key, &storage), &storage);
  EXPECT_TRUE(release(&storage));
}

TEST_F(QnnExecuTorchBackendTest, DestroyReleasesLastManagerOutsideCacheMutex) {
  int storage;
  size_t destroyed = 0;
  std::shared_ptr<void> manager(&storage, [&](void*) {
    ++destroyed;
    EXPECT_FALSE(contains(&storage));
  });
  DelegateKey key{42, nullptr};
  ASSERT_EQ(publish(key, &storage, manager), &storage);
  ASSERT_EQ(acquire(key), &storage);
  manager.reset();

  backend_.destroy(&storage);
  EXPECT_EQ(destroyed, 0);
  EXPECT_TRUE(contains(&storage));
  backend_.destroy(&storage);
  EXPECT_EQ(destroyed, 1);
  EXPECT_FALSE(contains(&storage));
  backend_.destroy(&storage);
  EXPECT_EQ(destroyed, 1);
}

TEST_F(QnnExecuTorchBackendTest, IsolatesAllocatorsWithIdenticalSignatures) {
  std::array<int, 2> storage{};
  runtime::MemoryAllocator first_allocator(0, nullptr);
  runtime::MemoryAllocator second_allocator(0, nullptr);
  DelegateKey first_key{42, &first_allocator};
  DelegateKey second_key{42, &second_allocator};
  ASSERT_EQ(publish(first_key, &storage[0]), &storage[0]);
  ASSERT_EQ(publish(second_key, &storage[1]), &storage[1]);
  EXPECT_TRUE(release(&storage[0]));
  EXPECT_FALSE(contains(&storage[0]));
  EXPECT_TRUE(contains(&storage[1]));
  EXPECT_EQ(acquire(second_key), &storage[1]);
  EXPECT_FALSE(release(&storage[1]));
  EXPECT_TRUE(release(&storage[1]));
}

TEST_F(QnnExecuTorchBackendTest, ConcurrentPublicationKeepsOneManager) {
  constexpr size_t count = 16;
  std::array<int, count> storage{};
  std::array<void*, count> published{};
  std::vector<std::thread> threads;
  threads.reserve(count);
  DelegateKey key{42, nullptr};
  for (size_t i = 0; i < count; ++i) {
    threads.emplace_back([&, i] { published[i] = publish(key, &storage[i]); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  void* winner = published[0];
  ASSERT_NE(winner, nullptr);
  for (size_t i = 0; i < count; ++i) {
    EXPECT_EQ(published[i], winner);
    EXPECT_EQ(contains(&storage[i]), &storage[i] == winner);
    EXPECT_EQ(release(winner), i == count - 1);
  }
  EXPECT_EQ(acquire(key), std::nullopt);
  EXPECT_FALSE(contains(winner));
}

TEST_F(QnnExecuTorchBackendTest, SpillFillOwnerOutlivesAllGroupMembers) {
  for (bool owner_first : {true, false}) {
    std::array<int, 2> storage{};
    std::vector<int> destroyed;
    auto group = spill_fill_group({nullptr, 1024});
    std::shared_ptr<void> owner(
        &storage[0], [&](void*) { destroyed.push_back(0); });
    std::shared_ptr<void> member(
        &storage[1], [&](void*) { destroyed.push_back(1); });
    group->owner = owner;
    ASSERT_EQ(publish({42, nullptr}, owner.get(), owner, group), &storage[0]);
    ASSERT_EQ(publish({43, nullptr}, member.get(), member, group), &storage[1]);
    std::weak_ptr<SpillFillGroup> previous_group = group;
    owner.reset();
    member.reset();
    group.reset();

    ASSERT_TRUE(release(&storage[owner_first ? 0 : 1]));
    EXPECT_EQ(
        destroyed, owner_first ? std::vector<int>{} : std::vector<int>{1});
    auto live_group = spill_fill_group({nullptr, 1024});
    EXPECT_EQ(live_group->owner.get(), &storage[0]);
    live_group.reset();
    ASSERT_TRUE(release(&storage[owner_first ? 1 : 0]));
    EXPECT_EQ(destroyed, (std::vector<int>{1, 0}));
    EXPECT_TRUE(previous_group.expired());
    EXPECT_EQ(spill_fill_group({nullptr, 1024})->owner, nullptr);
  }
}

TEST_F(QnnExecuTorchBackendTest, SpillFillGroupsAreIndependentAcrossModules) {
  runtime::MemoryAllocator first_allocator(0, nullptr);
  runtime::MemoryAllocator second_allocator(0, nullptr);
  std::array<int, 2> storage{};
  std::vector<int> destroyed;
  auto first_group = spill_fill_group({&first_allocator, 1024});
  auto second_group = spill_fill_group({&second_allocator, 1024});
  EXPECT_NE(first_group, second_group);
  EXPECT_NE(first_group, spill_fill_group({&first_allocator, 2048}));
  std::shared_ptr<void> first(
      &storage[0], [&](void*) { destroyed.push_back(0); });
  std::shared_ptr<void> second(
      &storage[1], [&](void*) { destroyed.push_back(1); });
  first_group->owner = first;
  second_group->owner = second;
  ASSERT_EQ(
      publish({42, &first_allocator}, first.get(), first, first_group),
      &storage[0]);
  ASSERT_EQ(
      publish({42, &second_allocator}, second.get(), second, second_group),
      &storage[1]);
  first.reset();
  second.reset();
  first_group.reset();
  second_group.reset();
  ASSERT_TRUE(release(&storage[0]));
  EXPECT_EQ(destroyed, (std::vector<int>{0}));
  EXPECT_TRUE(contains(&storage[1]));
  EXPECT_EQ(
      spill_fill_group({&second_allocator, 1024})->owner.get(), &storage[1]);
  ASSERT_TRUE(release(&storage[1]));
  EXPECT_EQ(destroyed, (std::vector<int>{0, 1}));
}

TEST_F(QnnExecuTorchBackendTest, ConcurrentLoadsAcquireOneSpillFillGroup) {
  constexpr size_t count = 16;
  std::array<std::shared_ptr<SpillFillGroup>, count> groups;
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    threads.emplace_back(
        [&, i] { groups[i] = spill_fill_group({nullptr, 1024}); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& group : groups) {
    EXPECT_EQ(group, groups[0]);
  }
}

} // namespace executorch::backends::qnn

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUDevice.h>

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <thread>
#include <vector>

namespace executorch::backends::webgpu {
namespace {

class ExplicitDefaultContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    set_default_webgpu_context(nullptr);
  }

  void TearDown() override {
    set_default_webgpu_context(nullptr);
  }
};

TEST_F(ExplicitDefaultContextTest, ClaimAndReleaseRequireExactOwner) {
  WebGPUContext first;
  WebGPUContext second;

  EXPECT_EQ(get_explicit_default_webgpu_context(), nullptr);
  EXPECT_TRUE(compare_and_set_default_webgpu_context(nullptr, &first));
  EXPECT_EQ(get_explicit_default_webgpu_context(), &first);
  EXPECT_EQ(get_default_webgpu_context(), &first);

  EXPECT_FALSE(compare_and_set_default_webgpu_context(nullptr, &second));
  EXPECT_FALSE(compare_and_set_default_webgpu_context(&second, nullptr));
  EXPECT_EQ(get_explicit_default_webgpu_context(), &first);

  EXPECT_TRUE(compare_and_set_default_webgpu_context(&first, nullptr));
  EXPECT_EQ(get_explicit_default_webgpu_context(), nullptr);
}

TEST_F(ExplicitDefaultContextTest, ConcurrentClaimsHaveOneWinner) {
  constexpr size_t kClaimants = 8;
  std::array<WebGPUContext, kClaimants> contexts;
  std::array<std::atomic<bool>, kClaimants> claimed;
  for (auto& value : claimed) {
    value.store(false);
  }

  std::vector<std::thread> threads;
  threads.reserve(kClaimants);
  for (size_t i = 0; i < kClaimants; i++) {
    threads.emplace_back([&, i]() {
      claimed[i].store(
          compare_and_set_default_webgpu_context(nullptr, &contexts[i]));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  size_t winner = kClaimants;
  size_t winners = 0;
  for (size_t i = 0; i < kClaimants; i++) {
    if (claimed[i].load()) {
      winner = i;
      winners++;
    }
  }
  ASSERT_EQ(winners, 1u);
  ASSERT_LT(winner, kClaimants);
  EXPECT_EQ(get_explicit_default_webgpu_context(), &contexts[winner]);

  for (size_t i = 0; i < kClaimants; i++) {
    if (i != winner) {
      EXPECT_FALSE(
          compare_and_set_default_webgpu_context(&contexts[i], nullptr));
    }
  }
  EXPECT_TRUE(
      compare_and_set_default_webgpu_context(&contexts[winner], nullptr));
  EXPECT_EQ(get_explicit_default_webgpu_context(), nullptr);
}

} // namespace
} // namespace executorch::backends::webgpu

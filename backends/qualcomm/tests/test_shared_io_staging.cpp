/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/SharedIoStaging.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

using executorch::backends::qnn::NeedsStaging;
using executorch::backends::qnn::PackStaging;
using executorch::backends::qnn::StagingBlocks;

TEST(SharedIoStaging, StagesOnlyMixedExecutes) {
  EXPECT_FALSE(NeedsStaging(0, 3));
  EXPECT_FALSE(NeedsStaging(3, 3));
  EXPECT_TRUE(NeedsStaging(1, 3));
  EXPECT_TRUE(NeedsStaging(2, 3));
  EXPECT_FALSE(NeedsStaging(0, 0));
}

TEST(SharedIoStaging, PacksAGraphsTensorsIntoOneAlignedBlock) {
  std::vector<size_t> offsets;
  EXPECT_EQ(PackStaging({3, 16, 1}, 16, offsets), 33u);
  EXPECT_EQ(offsets, (std::vector<size_t>{0, 16, 32}));
  EXPECT_EQ(PackStaging({}, 16, offsets), 0u);
  EXPECT_TRUE(offsets.empty());
}

TEST(SharedIoStaging, KeepsOneBlockPerGraphAndFreesOnDestruction) {
  std::vector<size_t> allocated;
  std::vector<void*> freed;
  void* first = nullptr;
  {
    StagingBlocks blocks(
        [&](size_t nbytes) {
          allocated.push_back(nbytes);
          return std::malloc(nbytes);
        },
        [&](void* data) {
          freed.push_back(data);
          std::free(data);
        });
    first = blocks.Get("prefill", 16);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(blocks.Get("prefill", 16), first);
    void* second = blocks.Get("decode", 32);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(second, first);
    EXPECT_EQ(blocks.Get("prefill", 8), nullptr);
    EXPECT_EQ(blocks.size(), 2u);
    EXPECT_TRUE(freed.empty());
  }
  EXPECT_EQ(allocated, (std::vector<size_t>{16, 32}));
  EXPECT_EQ(freed.size(), 2u);
  EXPECT_NE(std::find(freed.begin(), freed.end(), first), freed.end());
}

TEST(SharedIoStaging, FailedAllocationIsRetriedAndNotFreed) {
  int calls = 0, freed = 0;
  {
    StagingBlocks blocks(
        [&](size_t) -> void* {
          ++calls;
          return nullptr;
        },
        [&](void*) { ++freed; });
    EXPECT_EQ(blocks.Get("decode", 4), nullptr);
    EXPECT_EQ(blocks.Get("decode", 4), nullptr);
    EXPECT_EQ(blocks.size(), 0u);
  }
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(freed, 0);
}

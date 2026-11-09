// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/vulkan/VulkanConstantMaterializationTracker.h>

#include <stdexcept>

#include <gtest/gtest.h>

namespace ptn {
namespace {

// cppcheck-suppress-begin syntaxError
TEST(VulkanConstantMaterializationTrackerTest, UsesPackageAndOwnerIdentity) {
  VulkanConstantMaterializationTracker tracker;

  EXPECT_FALSE(tracker.record(1, "weight", 4));
  EXPECT_TRUE(tracker.record(1, "weight", 4));
  EXPECT_FALSE(tracker.record(2, "weight", 4));
  EXPECT_FALSE(tracker.record(1, "bias", 2));

  EXPECT_EQ(tracker.num_constants(), 3);
  EXPECT_EQ(tracker.unique_constant_bytes(), 10);
  EXPECT_EQ(tracker.materialized_constant_bytes(), 14);
}

TEST(VulkanConstantMaterializationTrackerTest, RejectsInconsistentMetadata) {
  VulkanConstantMaterializationTracker tracker;
  EXPECT_FALSE(tracker.record(1, "weight", 4));
  try {
    tracker.record(1, "weight", 8);
    FAIL() << "expected inconsistent metadata to be rejected";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "vulkan constant tracker: byte count changed for package 1 constant "
        "'weight': "
        "recorded 4, got 8");
  }
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn

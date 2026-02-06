/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/version.h>

#include <gtest/gtest.h>

#include <cstring>

using namespace ::testing;

TEST(VersionTest, VersionMacrosDefined) {
  // Verify that all version macros are defined and have valid values.
  EXPECT_GE(ET_VERSION_MAJOR, 0);
  EXPECT_GE(ET_VERSION_MINOR, 0);
  EXPECT_GE(ET_VERSION_PATCH, 0);
}

TEST(VersionTest, VersionCode) {
  // Verify that ET_VERSION_CODE is computed correctly from components.
  int expected_code =
      (ET_VERSION_MAJOR << 16) | (ET_VERSION_MINOR << 8) | ET_VERSION_PATCH;
  EXPECT_EQ(ET_VERSION_CODE, expected_code);
}

TEST(VersionTest, VersionCheck) {
  // Verify that ET_VERSION_CHECK computes the same value as ET_VERSION_CODE
  // for the current version.
  EXPECT_EQ(
      ET_VERSION_CODE,
      ET_VERSION_CHECK(ET_VERSION_MAJOR, ET_VERSION_MINOR, ET_VERSION_PATCH));

  // Verify version comparison logic works.
  // Current version should be >= 0.0.0.
  EXPECT_GE(ET_VERSION_CODE, ET_VERSION_CHECK(0, 0, 0));

  // Verify the bit layout: major has highest priority.
  EXPECT_GT(ET_VERSION_CHECK(1, 0, 0), ET_VERSION_CHECK(0, 255, 255));
  EXPECT_GT(ET_VERSION_CHECK(0, 1, 0), ET_VERSION_CHECK(0, 0, 255));
}

TEST(VersionTest, VersionComponentsMatchString) {
  // Build a version string from the components and verify it matches
  // ET_VERSION.
  char expected_version[32];
  std::snprintf(
      expected_version,
      sizeof(expected_version),
      "%d.%d.%d",
      ET_VERSION_MAJOR,
      ET_VERSION_MINOR,
      ET_VERSION_PATCH);
  EXPECT_STREQ(ET_VERSION, expected_version);
}

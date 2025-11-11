/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/platform.h>
#include <executorch/test/utils/DeathTest.h>

TEST(ExecutorPalTest, UninitializedPalDeath) {
  // Check for assertion failure on debug builds.

#ifndef NDEBUG

  ET_EXPECT_DEATH({ et_pal_current_ticks(); }, "");

  ET_EXPECT_DEATH(
      {
        et_pal_emit_log_message(
            0, et_pal_log_level_t::kFatal, "", "", 0, "", 0);
      },
      "");

#endif // !defined(NDEBUG)
}

/// Override the default weak main declaration.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Purposefully do not initialize the PAL.
  return RUN_ALL_TESTS();
}

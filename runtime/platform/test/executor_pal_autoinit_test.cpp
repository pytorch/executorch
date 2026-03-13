/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/runtime/platform/platform.h>

TEST(ExecutorPalTest, AutoInitialization) {
  // Functions should auto-initialize and work without explicit et_pal_init()
  et_timestamp_t time = et_pal_current_ticks();
  EXPECT_GE(time, 0);

  // Logging should also work
  et_pal_emit_log_message(
      0, et_pal_log_level_t::kInfo, "test", "test", 0, "auto-init test", 14);
}

/// Override the default weak main declaration.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Purposefully do not initialize the PAL.
  return RUN_ALL_TESTS();
}

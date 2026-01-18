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

TEST(VersionTest, ExecutorchVersionIsDefined) {
  EXPECT_NE(EXECUTORCH_VERSION, nullptr);
  EXPECT_NE(strlen(EXECUTORCH_VERSION), 0);
}

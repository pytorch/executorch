/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Custom main function for unit test executables.
 * Based on `fbcode//common/gtest/LightMain.cpp`
 */

#include <executorch/runtime/platform/compiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

// Define main as a weak symbol to allow trivial overriding.
int main(int argc, char** argv) __ET_WEAK;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::init(&argc, &argv);
  torch::executor::runtime_init();
  return RUN_ALL_TESTS();
}

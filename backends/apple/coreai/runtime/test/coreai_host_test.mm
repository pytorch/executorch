/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  @autoreleasepool {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
}

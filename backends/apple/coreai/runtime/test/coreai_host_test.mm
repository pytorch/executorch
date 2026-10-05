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
#include "coreai_acquisition_fixture.h"
#include "coreai_bookmark_fixture.h"

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  @autoreleasepool {
    executorch::backends::coreai::testing::set_acquisition_test_executable(argv[0]);
    const int acquisition_result =
        executorch::backends::coreai::testing::acquisition_child_mode(argc, argv);
    if (acquisition_result >= 0) return acquisition_result;
    executorch::backends::coreai::testing::set_bookmark_test_executable(argv[0]);
    const int child_result = executorch::backends::coreai::testing::bookmark_child_mode(argc, argv);
    if (child_result >= 0) return child_result;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
}

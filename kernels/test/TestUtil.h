/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Kernel Test utilities.
 */

#pragma once

#include <executorch/test/utils/DeathTest.h>
#include <gtest/gtest.h>

#ifdef USE_ATEN_LIB
/**
 * Ensure the kernel will fail when `_statement` is executed.
 * @param _statement Statement to execute.
 */
#define ET_EXPECT_KERNEL_FAILURE(_statement) EXPECT_ANY_THROW(_statement)

#define ET_EXPECT_KERNEL_FAILURE_WITH_MSG(_statement, _matcher) \
  EXPECT_ANY_THROW(_statement)

#else

#define ET_EXPECT_KERNEL_FAILURE(_statement) ET_EXPECT_DEATH(_statement, "")

#define ET_EXPECT_KERNEL_FAILURE_WITH_MSG(_statement, _matcher) \
  ET_EXPECT_DEATH(_statement, _matcher)

#endif // USE_ATEN_LIB

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Death test utility macros.
 */

#pragma once

#include <gtest/gtest.h>

#if ET_BUILD_MODE_COV

/**
 * TODO(T124640221): Work around a seg fault when running death tests in
 * @mode/dbgo-cov. If the root cause of that bug is fixed, remove this
 * ET_EXPECT_DEATH wrapper and go back to calling EXPECT_DEATH directly from
 * tests.
 */
#define ET_EXPECT_DEATH(_statement, _matcher) ((void)0)

#else // ET_BUILD_MODE_COV

/**
 * Ensure the executable will abort when `_statement` is executed.
 *
 * @param _statement Statement to execute.
 * @param _matcher Regex or matcher to match against the stderr output of
 * the dying process. If this does not match, the test will fail.
 */
#define ET_EXPECT_DEATH(_statement, _matcher) \
  EXPECT_DEATH_IF_SUPPORTED(_statement, _matcher)

#endif // ET_BUILD_MODE_COV

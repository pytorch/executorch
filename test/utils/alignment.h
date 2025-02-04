/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cinttypes>

#include <gmock/gmock.h> // For MATCHER_P

namespace executorch {
namespace runtime {
namespace testing {

/**
 * Returns true if the address of `ptr` is a whole multiple of `alignment`.
 */
inline bool is_aligned(const void* ptr, size_t alignment) {
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  return addr % alignment == 0;
}

/**
 * Lets gtest users write `EXPECT_THAT(ptr, IsAlignedTo(alignment))` or
 * `EXPECT_THAT(ptr, Not(IsAlignedTo(alignment)))`.
 *
 * See also `EXPECT_ALIGNED()`.
 */
MATCHER_P(IsAlignedTo, other, "") {
  return is_aligned(arg, other);
}

/*
 * Helpers for checking the alignment of a pointer.
 */

#define EXPECT_ALIGNED(ptr, alignment) \
  EXPECT_THAT((ptr), executorch::runtime::testing::IsAlignedTo((alignment)))
#define ASSERT_ALIGNED(ptr, alignment) \
  ASSERT_THAT((ptr), executorch::runtime::testing::IsAlignedTo((alignment)))

} // namespace testing
} // namespace runtime
} // namespace executorch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// `ET_SKIP_IF(cond, reason)` -- skip a kernel test when `cond` is true.
//
// Replaces the older inline pattern:
//     if (SupportedFeatures::get()->is_aten) {
//       GTEST_SKIP() << "ATen handles X";
//     }
// with:
//     ET_SKIP_IF(SupportedFeatures::get()->is_aten, "ATen handles X");
//
// OSS:    expands to `if (cond) GTEST_SKIP() << reason;` (unchanged).
// fbcode: expands to `if (cond) return;` so the test reports PASS, not SKIP.
//
// fbcode's TestX flags consistently-skipping tests as "broken" -- see
// T208053850 and
// https://fb.workplace.com/groups/testinfra.discuss/permalink/2044665472719153/.
// Collapse back to the OSS form once that's resolved.
//
// `EXECUTORCH_INTERNAL` is set by BUCK gated on `runtime.is_oss` (see
// `runtime/executor/targets.bzl` for the existing precedent).

#if defined(EXECUTORCH_INTERNAL) && EXECUTORCH_INTERNAL == 1

namespace executorch::testing::internal {
// No-op sink so `<<` chains in the reason still parse and type-check.
struct SkipReasonSink {
  template <typename T>
  const SkipReasonSink& operator<<(const T&) const {
    return *this;
  }
};
} // namespace executorch::testing::internal

// `if/else` form avoids dangling-else hazards and lets the reason still
// participate in `<<` chains.
#define ET_SKIP_IF(cond, reason) \
  if ((cond)) {                  \
    return;                      \
  } else                         \
    ::executorch::testing::internal::SkipReasonSink{} << reason

#else // !EXECUTORCH_INTERNAL

#include <gtest/gtest.h>

#define ET_SKIP_IF(cond, reason) \
  if ((cond))                    \
  GTEST_SKIP() << reason

#endif // EXECUTORCH_INTERNAL

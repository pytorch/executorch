/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file
 * Clock and timing related methods.
 */

#pragma once

#include <executorch/runtime/platform/platform.h>

namespace torch {
namespace executor {

/**
 * Convert an interval from units of system ticks to nanoseconds.
 * The conversion ratio is platform-dependent, and thus depends on
 * the platform implementation of et_pal_ticks_to_ns_multiplier().
 *
 * @param[in] ticks The interval length in system ticks.
 * @retval The interval length in nanoseconds.
 */
inline uint64_t ticks_to_ns(et_timestamp_t ticks) {
  et_tick_ratio_t ratio = et_pal_ticks_to_ns_multiplier();
  return static_cast<uint64_t>(ticks) * ratio.numerator / ratio.denominator;
}

} // namespace executor
} // namespace torch

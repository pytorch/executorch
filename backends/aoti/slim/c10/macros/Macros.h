/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// SlimTensor Macros Header
//
// This header provides common compiler hint macros for SlimTensor code.

// Compiler hint macros for branch prediction
#if defined(__GNUC__) || defined(__clang__)
#define SLIMTENSOR_LIKELY(x) __builtin_expect(!!(x), 1)
#define SLIMTENSOR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define SLIMTENSOR_LIKELY(x) (x)
#define SLIMTENSOR_UNLIKELY(x) (x)
#endif

namespace executorch::backends::aoti::slim::c10 {
// Empty namespace - macros are defined above
} // namespace executorch::backends::aoti::slim::c10

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Thin wrapper to reuse ExecuTorch's c10 TypeSafeSignMath implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::{is_negative, signum, signs_differ,
// greater_than_max, less_than_lowest}.

#include <c10/util/TypeSafeSignMath.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {

using ::c10::greater_than_max;
using ::c10::is_negative;
using ::c10::less_than_lowest;
using ::c10::signs_differ;
using ::c10::signum;

} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch

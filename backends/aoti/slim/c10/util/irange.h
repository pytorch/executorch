/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Thin wrapper to reuse ExecuTorch's c10::irange implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::irange.

#include <c10/util/irange.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {

using ::c10::irange;

} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch

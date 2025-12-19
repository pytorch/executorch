/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Thin wrapper to reuse ExecuTorch's c10 floating_point_utils implementation.
// This provides backward compatibility for SlimTensor code that uses
// executorch::backends::aoti::slim::c10::detail::{fp32_from_bits,
// fp32_to_bits}.

#include <c10/util/floating_point_utils.h>

namespace executorch {
namespace backends {
namespace aoti {
namespace slim {
namespace c10 {
namespace detail {

using ::c10::detail::fp32_from_bits;
using ::c10::detail::fp32_to_bits;

} // namespace detail
} // namespace c10
} // namespace slim
} // namespace aoti
} // namespace backends
} // namespace executorch

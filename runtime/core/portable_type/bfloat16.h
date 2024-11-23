/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/BFloat16.h>

namespace executorch::runtime::etensor {
using c10::BFloat16;
namespace internal {
using c10::detail::f32_from_bits;
using c10::detail::round_to_nearest_even;
} // namespace internal
} // namespace executorch::runtime::etensor

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::BFloat16;
} // namespace executor
} // namespace torch

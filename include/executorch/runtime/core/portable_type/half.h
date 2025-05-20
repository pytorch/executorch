/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <c10/util/Half.h>

namespace executorch::runtime::etensor {
using c10::Half;
namespace internal {
using c10::detail::fp16_ieee_from_fp32_value;
using c10::detail::fp16_ieee_to_fp32_bits;
using c10::detail::fp16_ieee_to_fp32_value;
using c10::detail::fp32_from_bits;
using c10::detail::fp32_to_bits;
} // namespace internal
} // namespace executorch::runtime::etensor
namespace torch::executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::etensor::Half;
} // namespace torch::executor

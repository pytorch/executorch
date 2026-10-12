/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>

#include <executorch/runtime/core/exec_aten/exec_aten.h>

// A symmetric weight has a zero point of zero by definition, so the AoT side
// omits the argument entirely rather than materialising a constant tensor of
// zeros - one int32 per output channel that never carries any information.
// Absent therefore means zero, and only an asymmetric weight supplies a value.

namespace impl::generic::quantized {

inline int32_t resolve_weight_zero_point(
    const std::optional<::executorch::aten::Tensor>& weight_zero_point) {
  return weight_zero_point.has_value()
      ? weight_zero_point->const_data_ptr<int32_t>()[0]
      : 0;
}

struct PerChannelWeightZeroPoint {
  const int32_t* data;
  int32_t stride;
};

// When absent, a stride of zero makes every channel read the same zero.
inline PerChannelWeightZeroPoint resolve_per_channel_weight_zero_point(
    const std::optional<::executorch::aten::Tensor>& weight_zero_point) {
  static constexpr int32_t kZero = 0;
  if (!weight_zero_point.has_value()) {
    return {&kZero, 0};
  }
  return {
      weight_zero_point->const_data_ptr<int32_t>(),
      weight_zero_point->numel() > 1 ? 1 : 0};
}

} // namespace impl::generic::quantized

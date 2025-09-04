/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace vkcompute {

enum class QuantizationGranularity {
  PerChannel,
  PerTensor,
  PerGroup,
  None,
};

static constexpr QuantizationGranularity kPerChannel =
    QuantizationGranularity::PerChannel;
static constexpr QuantizationGranularity kPerTensor =
    QuantizationGranularity::PerTensor;
static constexpr QuantizationGranularity kPerGroup =
    QuantizationGranularity::PerGroup;
static constexpr QuantizationGranularity kNoQuantization =
    QuantizationGranularity::None;

struct QuantizationConfig {
  int nbits;
  QuantizationGranularity granularity;
  std::vector<int64_t> granularity_sizes;
  bool is_symmetric;
  bool is_dynamic;

  QuantizationConfig()
      : nbits(8),
        granularity(kPerTensor),
        granularity_sizes(),
        is_symmetric(true),
        is_dynamic(false) {}

  QuantizationConfig(
      int nbits_,
      QuantizationGranularity granularity_,
      const std::vector<int64_t>& granularity_sizes_,
      bool is_symmetric_ = true,
      bool is_dynamic_ = false)
      : nbits(nbits_),
        granularity(granularity_),
        granularity_sizes(granularity_sizes_),
        is_symmetric(is_symmetric_),
        is_dynamic(is_dynamic_) {}
};

} // namespace vkcompute

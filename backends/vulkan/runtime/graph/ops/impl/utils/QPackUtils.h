/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

std::vector<uint8_t> int4mm_pack_weights(
    const std::vector<int64_t>& W_sizes,
    const uint8_t* w_ptr);

std::vector<float> int4mm_dequantize_weights(
    const std::vector<int64_t>& W_sizes,
    const uint8_t* w_ptr,
    const uint32_t group_size,
    const float* scales_and_zeros);

} // namespace vkcompute

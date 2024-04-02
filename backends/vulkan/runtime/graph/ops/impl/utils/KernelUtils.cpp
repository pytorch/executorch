/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

int64_t calc_out_size(
    const int64_t in_size,
    const int64_t kernel,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const bool ceil_mode) {
  int64_t c = ceil_mode ? stride - 1 : 0;
  int64_t out_size =
      (in_size + 2 * padding - dilation * (kernel - 1) - 1 + c) / stride + 1;
  if (ceil_mode && (out_size - 1) * stride >= in_size + padding) {
    --out_size;
  }
  return out_size;
}

api::utils::ivec2 reverse(ComputeGraph& graph, ValueRef vref) {
  return api::utils::make_ivec2(
      graph.get_val(vref).toIntList(), /*reverse=*/true);
}

} // namespace vkcompute

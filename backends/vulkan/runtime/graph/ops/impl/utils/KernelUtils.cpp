/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

api::utils::ivec2
make_ivec2_int_list(ComputeGraph& graph, ValueRef vref, const bool reverse) {
  return api::utils::make_ivec2(graph.get_val(vref).toIntList(), reverse);
}

KernelParams create_kernel_params(
    ComputeGraph& graph,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation) {
  return {
      make_ivec2_int_list(graph, kernel_size, /*reverse=*/true),
      make_ivec2_int_list(graph, stride, /*reverse=*/true),
      make_ivec2_int_list(graph, padding, /*reverse=*/true),
      make_ivec2_int_list(graph, dilation, /*reverse=*/true),
  };
}

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

std::vector<int64_t> calc_hw_out_sizes(
    ComputeGraph& graph,
    const std::vector<int64_t>& in_sizes,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef ceil_mode) {
  const int64_t ndim = in_sizes.size();
  std::vector<int64_t> out_sizes(2);

  const auto kernel_vec =
      make_ivec2_int_list(graph, kernel_size, /*reverse=*/false);
  const auto stride_vec = make_ivec2_int_list(graph, stride, /*reverse=*/false);
  const auto padding_vec =
      make_ivec2_int_list(graph, padding, /*reverse=*/false);
  const auto dilation_vec =
      make_ivec2_int_list(graph, dilation, /*reverse=*/false);

  // Height
  out_sizes.at(0) = calc_out_size(
      in_sizes.at(ndim - 2),
      kernel_vec.data[0],
      stride_vec.data[0],
      padding_vec.data[0],
      dilation_vec.data[0],
      ceil_mode);
  // Width
  out_sizes.at(1) = calc_out_size(
      in_sizes.at(ndim - 1),
      kernel_vec.data[1],
      stride_vec.data[1],
      padding_vec.data[1],
      dilation_vec.data[1],
      ceil_mode);

  VK_CHECK_COND(out_sizes.at(0) >= 1);
  VK_CHECK_COND(out_sizes.at(1) >= 1);

  return out_sizes;
}

} // namespace vkcompute

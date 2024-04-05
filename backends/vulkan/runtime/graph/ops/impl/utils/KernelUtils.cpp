/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

api::utils::ivec2 make_ivec2_from_list(ComputeGraph& graph, ValueRef vref) {
  return api::utils::make_ivec2(
      graph.get_val(vref).toIntList(), /*reverse = */ true);
}

api::utils::ivec2 make_ivec2_kernel_size(
    ComputeGraph& graph,
    const ValueRef weight,
    const bool kernel_size_only) {
  if (kernel_size_only) {
    return make_ivec2_from_list(graph, weight);
  } else {
    const auto weight_sizes = graph.get_val(weight).toTensorRef().sizes;
    return api::utils::make_ivec2({weight_sizes.at(3), weight_sizes.at(2)});
  }
}

KernelParams create_kernel_params(
    ComputeGraph& graph,
    const ValueRef weight,
    const bool kernel_size_only,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation) {
  return {
      make_ivec2_kernel_size(graph, weight, kernel_size_only),
      make_ivec2_from_list(graph, stride),
      make_ivec2_from_list(graph, padding),
      make_ivec2_from_list(graph, dilation),
  };
}

int64_t calc_out_size(
    const int64_t in_size,
    const int64_t kernel_size,
    const int64_t stride,
    const int64_t padding,
    const int64_t dilation,
    const bool ceil_mode) {
  int64_t c = ceil_mode ? stride - 1 : 0;
  int64_t out_size =
      (in_size + 2 * padding - dilation * (kernel_size - 1) - 1 + c) / stride +
      1;
  if (ceil_mode && (out_size - 1) * stride >= in_size + padding) {
    --out_size;
  }
  return out_size;
}

std::vector<int64_t> calc_out_sizes_hw(
    ComputeGraph& graph,
    const std::vector<int64_t>& in_sizes,
    const ValueRef weight,
    const bool kernel_size_only,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef ceil_mode) {
  const int64_t ndim = in_sizes.size();
  std::vector<int64_t> out_sizes(2);

  const auto kernel_vec =
      make_ivec2_kernel_size(graph, weight, kernel_size_only);
  const auto stride_vec = make_ivec2_from_list(graph, stride);
  const auto padding_vec = make_ivec2_from_list(graph, padding);
  const auto dilation_vec = make_ivec2_from_list(graph, dilation);
  const bool ceil_mode_val =
      ceil_mode == kDummyValueRef ? false : graph.get_val(ceil_mode).toBool();

  // Height
  out_sizes.at(0) = calc_out_size(
      in_sizes.at(ndim - 2),
      kernel_vec.data[1],
      stride_vec.data[1],
      padding_vec.data[1],
      dilation_vec.data[1],
      ceil_mode_val);
  // Width
  out_sizes.at(1) = calc_out_size(
      in_sizes.at(ndim - 1),
      kernel_vec.data[0],
      stride_vec.data[0],
      padding_vec.data[0],
      dilation_vec.data[0],
      ceil_mode_val);

  VK_CHECK_COND(out_sizes.at(0) >= 1);
  VK_CHECK_COND(out_sizes.at(1) >= 1);

  return out_sizes;
}

} // namespace vkcompute

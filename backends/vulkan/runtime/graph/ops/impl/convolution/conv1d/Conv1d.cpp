/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv1d/Conv1dDirect.h>

namespace vkcompute {

void conv1d_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  if (graph.packed_dim_of(args[0]) == WHCN::kHeightDim) {
    // Height-packed: route to optimized conv1d implementations
    const auto weight_sizes = graph.sizes_of(args[1]);
    const int64_t groups_val = graph.get_int(args[8]);
    const bool is_pointwise = weight_sizes.at(2) == 1;
    const bool is_depthwise =
        groups_val == weight_sizes.at(0) && weight_sizes.at(1) == 1;

    // Build unified 10-arg vector:
    //   in, weight, bias, stride, padding, dilation, groups,
    //   output_min, output_max, out
    // For non-clamp (args.size() == 10): output_min/max = kDummyValueRef
    // For clamp (args.size() == 12): output_min/max from args[9]/args[10]
    ValueRef output_min = kDummyValueRef;
    ValueRef output_max = kDummyValueRef;
    ValueRef out;
    if (args.size() == 10) {
      out = args[9];
    } else {
      output_min = args[9];
      output_max = args[10];
      out = args[11];
    }

    std::vector<ValueRef> conv1d_args = {
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[8],
        output_min,
        output_max,
        out};

    if (is_pointwise) {
      VK_GET_OP_FN("et_vk.conv1d_pw.default")(graph, conv1d_args);
    } else if (is_depthwise) {
      VK_GET_OP_FN("et_vk.conv1d_dw.default")(graph, conv1d_args);
    } else {
      VK_THROW(
          "Height-packed conv1d only supports pointwise (K=1) or "
          "depthwise (groups=C)");
    }
    return;
  }

  // Existing channels-packed fallback
  if (args.size() == 10) {
    // ordinary conv1d
    return add_conv1d_direct_node(
        graph,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[8],
        /*out_min = */ kDummyValueRef,
        /*out_max = */ kDummyValueRef,
        args[9],
        false);
  } else {
    // conv1d with clamp
    return add_conv1d_direct_node(
        graph,
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        args[8],
        args[9],
        args[10],
        args[11],
        true);
  }
}

} // namespace vkcompute

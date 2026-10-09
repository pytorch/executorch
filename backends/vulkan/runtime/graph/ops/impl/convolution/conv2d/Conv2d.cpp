/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dDirect.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/im2col/Conv2dGemm.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  size_t ndim = graph->dim_of(self);
  std::vector<int64_t> new_out_sizes(ndim);
  const bool transposed = graph->get_bool(extra_args.at(4));

  std::vector<int64_t> self_sizes = graph->sizes_of(self);
  // Batch, Channel
  if (ndim == 4) {
    new_out_sizes.at(ndim - 4) = self_sizes.at(ndim - 4);
  }

  TensorRefPtr weight_ref = graph->get_tref(extra_args.at(0));
  const auto& weight_sizes = weight_ref->sizes;
  new_out_sizes.at(ndim - 3) =
      transposed ? weight_sizes.at(ndim - 3) : weight_sizes.at(ndim - 4);

  // Height, Width
  const auto& new_out_sizes_hw = calc_out_sizes_hw(
      *graph,
      self_sizes,
      extra_args.at(0),
      /*kernel_size_only = */ false,
      {extra_args.at(1), extra_args.at(2), extra_args.at(3), extra_args.at(5)},
      transposed);
  new_out_sizes.at(ndim - 2) = new_out_sizes_hw.at(0);
  new_out_sizes.at(ndim - 1) = new_out_sizes_hw.at(1);

  graph->virtual_resize(out, new_out_sizes);
}

Conv2dMethod get_conv2d_method(
    ComputeGraph& graph,
    const ValueRef weight,
    const int64_t groups,
    const bool transposed) {
  const auto weight_sizes = graph.sizes_of(weight);
  if (!transposed && weight_sizes.at(0) == groups && weight_sizes.at(1) == 1) {
    return Conv2dMethod::Depthwise;
  }
  if (transposed) {
    return Conv2dMethod::Transposed;
  }
  if (weight_sizes.at(2) == 1 && weight_sizes.at(3) == 1) {
    return Conv2dMethod::Pointwise;
  }
  return Conv2dMethod::SlidingWindow;
}

// Decide whether a SlidingWindow conv2d should be computed via the
// im2col + GEMM path (conv2d_gemm_impl) instead of the direct convolution
// shader. Across 26 configs on Mali-G715 (buffer path) and Adreno SM8650
// (texture path): FP32 cases were numerically verified against the reference;
// FP16 cases were routing/dispatch-validated only (the reference is float-only
// for the large shapes, so FP16 outputs were not numerically checked).
//
// Only called for SlidingWindow conv2d (1x1 is routed to conv2d_pw and
// Depthwise/Transposed are handled before the call site).
//
// Preconditions (fall back to direct conv if any fail — the im2col path is
// either not applicable or not beneficial):
//   - groups == 1
//   - dilation == 1 (all dims)
//
// Selection rule: use im2col once the output channel count is large enough to
// amortize the fixed ~N*K_total im2col gather cost.
constexpr int64_t kIm2colMinCOut = 128;

// A cheap gather is a second, independent reason to take im2col. The gather
// materializes an N x K_total matrix before the GEMM runs, and a large gather
// is fine when c_out is large because the GEMM reads it back c_out times. When
// c_out is small it is only worth paying if the matrix is small outright, which
// is where the direct shader on Mali loses badly: an 80x80 3x3 conv wants ~4M
// elements, a 640x640 9x9 conv wants ~100M.
constexpr int64_t kIm2colMaxCheapGatherElements = 32 * 1024 * 1024;

bool should_use_conv2d_im2col(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const int64_t groups_val,
    const Kernel2dParams& kernel_params,
    const ValueRef out) {
  if (groups_val != 1) {
    return false;
  }
  if (kernel_params.dilation[0] != 1 || kernel_params.dilation[1] != 1) {
    return false;
  }
  const auto weight_sizes = graph.sizes_of(weight_data);
  const int64_t c_out = weight_sizes.at(0);
  const int64_t k_total =
      weight_sizes.at(1) * weight_sizes.at(2) * weight_sizes.at(3);

  const auto out_sizes = graph.sizes_of(out);
  const size_t ndim = out_sizes.size();
  const int64_t n = out_sizes.at(ndim - 1) * out_sizes.at(ndim - 2);
  return c_out >= kIm2colMinCOut ||
      (graph.device_is_mali() && n * k_total <= kIm2colMaxCheapGatherElements);
}

void conv2d_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef in = args[0];
  const ValueRef weight_data = args[1];
  const ValueRef bias = args[2];
  const ValueRef stride = args[3];
  const ValueRef padding = args[4];
  const ValueRef dilation = args[5];
  const ValueRef transposed = args[6];
  const ValueRef output_padding = args[7];
  const ValueRef groups = args[8];
  // conv2d with clamp passes out_min and out_max before the output
  const bool clamp_out = args.size() != 10;
  const ValueRef out_min = clamp_out ? args[9] : kDummyValueRef;
  const ValueRef out_max = clamp_out ? args[10] : kDummyValueRef;
  const ValueRef out = clamp_out ? args[11] : args[9];

  const bool transposed_val = graph.get_bool(transposed);

  float out_min_val = 0.0f;
  float out_max_val = 0.0f;
  if (out_min != kDummyValueRef) {
    out_min_val = graph.extract_scalar<float>(out_min);
  }
  if (out_max != kDummyValueRef) {
    out_max_val = graph.extract_scalar<float>(out_max);
  }

  const int64_t groups_val = graph.get_int(groups);

  const Conv2dMethod method =
      get_conv2d_method(graph, weight_data, groups_val, transposed_val);

  // Use tiled path for all pointwise conv2d
  if (method == Conv2dMethod::Pointwise) {
    return conv2d_pw_impl(
        graph,
        in,
        weight_data,
        bias,
        stride,
        padding,
        out,
        transposed_val,
        clamp_out,
        out_min_val,
        out_max_val);
  }

  if (method == Conv2dMethod::Depthwise) {
    return conv2d_dw_impl(
        graph,
        in,
        weight_data,
        bias,
        stride,
        padding,
        dilation,
        out,
        clamp_out,
        out_min_val,
        out_max_val);
  }

  const Kernel2dParams kernel_params = create_kernel2d_params(
      graph,
      weight_data,
      /*kernel_size_only = */ false,
      stride,
      padding,
      dilation);

  // SlidingWindow conv2d: route to the im2col + GEMM path when the heuristic
  // indicates it is beneficial, falling back to the direct convolution shader
  // otherwise.
  const bool use_im2col = method == Conv2dMethod::SlidingWindow &&
      should_use_conv2d_im2col(
          graph, weight_data, groups_val, kernel_params, out);
  if (use_im2col) {
    return conv2d_gemm_impl(
        graph,
        in,
        weight_data,
        bias,
        stride,
        padding,
        dilation,
        out,
        clamp_out,
        out_min_val,
        out_max_val);
  }

  return conv2d_direct_impl(
      graph,
      in,
      weight_data,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out,
      clamp_out,
      out_min_val,
      out_max_val);
}

} // namespace vkcompute

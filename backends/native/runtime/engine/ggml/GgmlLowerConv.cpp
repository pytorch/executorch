// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/engine/ggml/GgmlGraphBuilder.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <ggml.h>

#include <executorch/backends/native/runtime/Method.h>
#include <executorch/backends/native/runtime/engine/ggml/GgmlTypes.h>

namespace ptn::ggml_detail {

void GraphBuilder::lower_convolution(const Node& node) {
  const TensorRef input = tensor_arg(node, "input", 0);
  const TensorRef weight = tensor_arg(node, "weight", 1);
  const size_t rank = input.sizes.size();
  if (bool_arg(node, "transposed", 6, false) || (rank != 3 && rank != 4) ||
      weight.sizes.size() != rank) {
    throw std::runtime_error(
        "ggml: convolution supports 1D and 2D non-transposed");
  }
  require_f32(node, input.tensor);
  require_f32(node, weight.tensor);
  auto stride = pair_arg(node, "stride", 3, {1, 1}, 1);
  auto padding = pair_arg(node, "padding", 4, {0, 0}, 0);
  auto dilation = pair_arg(node, "dilation", 5, {1, 1}, 1);
  const int64_t groups = int_arg(node, "groups", 8, 1, true);
  const int64_t channels = input.sizes[1];
  const int64_t out_channels = weight.sizes[0];
  ggml_context* ctx = instance_.context.get();
  ggml_tensor* in = contiguous(input.tensor);
  ggml_tensor* kernel = contiguous(weight.tensor);
  if (rank == 3) {
    // [N, C, L] runs as a height-1 image.
    in = ggml_reshape_4d(
        ctx, in, input.sizes[2], 1, input.sizes[1], input.sizes[0]);
    kernel = ggml_reshape_4d(
        ctx, kernel, weight.sizes[2], 1, weight.sizes[1], weight.sizes[0]);
    stride[1] = 1;
    padding[1] = 0;
    dilation[1] = 1;
  }
  ggml_tensor* result = nullptr;
  if (groups == 1 && weight.sizes[1] == channels) {
    // ggml_conv_2d rounds the im2col columns of an F32 kernel through F16.
    ggml_tensor* columns = nullptr;
    if (kernel->ne[0] == 1 && kernel->ne[1] == 1) {
      // ggml-metal's im2col runs a threadgroup per output element with a
      // thread per kernel tap, one thread for a 1x1 kernel. With the
      // channels as im2col's batch they fill the threadgroups instead, and
      // a copy transposes them to the front; at stride 1 without padding
      // the copy alone suffices.
      const int64_t batch = in->ne[3];
      ggml_tensor* taps = in;
      if (stride != std::array<int, 2>{1, 1} ||
          padding != std::array<int, 2>{0, 0}) {
        taps = ggml_im2col(
            ctx,
            ggml_reshape_4d(ctx, kernel, 1, 1, 1, channels * out_channels),
            ggml_reshape_4d(ctx, in, in->ne[0], in->ne[1], 1, channels * batch),
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            /*is_2D=*/true,
            GGML_TYPE_F32);
        taps = ggml_reshape_4d(
            ctx, taps, taps->ne[1], taps->ne[2], channels, batch);
      }
      columns = ggml_cont(ctx, ggml_permute(ctx, taps, 1, 2, 0, 3));
    } else {
      columns = ggml_im2col(
          ctx,
          kernel,
          in,
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1],
          /*is_2D=*/true,
          GGML_TYPE_F32);
    }
    result = ggml_mul_mat(
        ctx,
        ggml_reshape_2d(
            ctx,
            columns,
            columns->ne[0],
            columns->ne[1] * columns->ne[2] * columns->ne[3]),
        ggml_reshape_2d(ctx, kernel, columns->ne[0], out_channels));
    result = ggml_reshape_4d(
        ctx,
        result,
        columns->ne[1],
        columns->ne[2],
        columns->ne[3],
        out_channels);
    result = materialize(ggml_permute(ctx, result, 0, 1, 3, 2));
  } else if (
      groups == channels && out_channels == channels && weight.sizes[1] == 1) {
    result = ggml_conv_2d_dw_direct(
        ctx,
        kernel,
        in,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1]);
  } else {
    throw std::runtime_error(
        "ggml: convolution supports groups of 1 or depthwise only");
  }
  std::optional<TensorRef> bias = optional_tensor(node, "bias", 2);
  if (bias && bias->sizes != std::vector<int64_t>{out_channels}) {
    throw std::runtime_error(
        "ggml: convolution bias must hold one value per output channel");
  }
  if (const auto found = batch_norms_.conv_bias.find(output_ids(node).at(0));
      found != batch_norms_.conv_bias.end()) {
    bias = TensorRef{found->second, {out_channels}};
  }
  if (bias) {
    result = ggml_add(
        ctx, result, ggml_reshape_4d(ctx, bias->tensor, 1, 1, out_channels, 1));
  }
  std::vector<int64_t> sizes{
      result->ne[3], result->ne[2], result->ne[1], result->ne[0]};
  if (rank == 3) {
    sizes.erase(sizes.begin() + 2);
    result = ggml_reshape_3d(ctx, result, sizes[2], sizes[1], sizes[0]);
  }
  set_single_output(node, TensorRef{result, std::move(sizes)});
}

// Only the normalized output is lowered; the saved statistics are empty in
// inference.
void GraphBuilder::lower_batch_norm(const Node& node) {
  const TensorRef input = tensor_arg(node, "input", 0);
  const std::optional<TensorRef> weight_ref =
      optional_tensor(node, "weight", 1);
  const std::optional<TensorRef> bias_ref = optional_tensor(node, "bias", 2);
  const TensorRef mean = tensor_arg(node, "running_mean", 3);
  const TensorRef var = tensor_arg(node, "running_var", 4);
  const double eps = float_arg(node, "eps", 6, 1e-5, true);
  if (input.sizes.size() < 2) {
    throw std::runtime_error("ggml: batch norm needs a channel dimension");
  }
  const std::vector<int64_t> per_channel{input.sizes[1]};
  if (mean.sizes != per_channel || var.sizes != per_channel ||
      (weight_ref && weight_ref->sizes != per_channel) ||
      (bias_ref && bias_ref->sizes != per_channel)) {
    throw std::runtime_error(
        "ggml: batch norm statistics must hold one value per channel");
  }
  require_only_first_output_used(node);
  ggml_tensor* weight = weight_ref ? weight_ref->tensor : nullptr;
  ggml_tensor* bias = bias_ref ? bias_ref->tensor : nullptr;
  ggml_context* ctx = instance_.context.get();
  const ValueId output = output_ids(node).at(0);
  if (batch_norms_.absorbed.contains(output)) {
    set(output, input);
    return;
  }
  ggml_tensor* scale = nullptr;
  ggml_tensor* shift = nullptr;
  if (const auto found = batch_norms_.affine.find(output);
      found != batch_norms_.affine.end()) {
    std::tie(scale, shift) = found->second;
  } else {
    // y = x * scale + shift with scale = weight / sqrt(var + eps) and
    // shift = bias - mean * scale.
    ggml_tensor* stddev = ggml_sqrt(
        ctx,
        ggml_scale_bias(
            ctx, as_f32(var.tensor), 1.0f, static_cast<float>(eps)));
    scale = ggml_div(
        ctx,
        weight != nullptr ? as_f32(weight)
                          : ggml_scale_bias(ctx, stddev, 0.0f, 1.0f),
        stddev);
    shift = ggml_mul(ctx, as_f32(mean.tensor), scale);
    shift = bias != nullptr ? ggml_sub(ctx, as_f32(bias), shift)
                            : ggml_neg(ctx, shift);
  }
  std::vector<int64_t> channel_sizes(input.sizes.size(), 1);
  channel_sizes[1] = input.sizes[1];
  ggml_tensor* result =
      ggml_mul(ctx, input.tensor, reshape(ctx, scale, channel_sizes));
  result = ggml_add(ctx, result, reshape(ctx, shift, channel_sizes));
  set(output, TensorRef{result, input.sizes});
}

// Indices are not produced; the lowering fails if they are consumed.
void GraphBuilder::lower_max_pool2d(const Node& node) {
  const TensorRef input = tensor_arg(node, "self", 0);
  if (input.sizes.size() != 4) {
    throw std::runtime_error("ggml: max_pool2d needs an NCHW input");
  }
  const auto kernel = pair_arg(node, "kernel_size", 1, {0, 0}, 1);
  const auto stride = pair_arg(node, "stride", 2, kernel, 1);
  const auto padding = pair_arg(node, "padding", 3, {0, 0}, 0);
  const auto dilation = pair_arg(node, "dilation", 4, {1, 1}, 1);
  require_only_first_output_used(node);
  if (kernel[0] <= 0 || kernel[1] <= 0 ||
      dilation != std::array<int, 2>{1, 1} ||
      bool_arg(node, "ceil_mode", 5, false)) {
    throw std::runtime_error(
        "ggml: max_pool2d supports dilation 1 without ceil_mode");
  }
  require_f32(node, input.tensor);
  ggml_tensor* result = ggml_pool_2d(
      instance_.context.get(),
      contiguous(input.tensor),
      GGML_OP_POOL_MAX,
      kernel[0],
      kernel[1],
      stride[0],
      stride[1],
      static_cast<float>(padding[0]),
      static_cast<float>(padding[1]));
  std::vector<int64_t> sizes{
      result->ne[3], result->ne[2], result->ne[1], result->ne[0]};
  set(output_ids(node).at(0), TensorRef{result, std::move(sizes)});
}

} // namespace ptn::ggml_detail

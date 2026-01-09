/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <tuple>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;
using OptIntArrayRef = executorch::aten::OptionalArrayRef<int64_t>;

namespace {

bool check_convolution_backward_args(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    ET_UNUSED const OptIntArrayRef bias_sizes_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    executorch::aten::ArrayRef<bool> output_mask,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  ET_CHECK_OR_RETURN_FALSE(
      transposed == false, "Transposed Convolution Backward not supported yet");
  ET_CHECK_OR_RETURN_FALSE(
      weight.dim() == 4,
      "Only 2D Convolution Backward supported for now; weight.dim() = %" ET_PRI_TENSOR_DIM,
      weight.dim());

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(weight, input));
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_output, input));

  if (output_mask[0]) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_input, input));
  }

  if (output_mask[1]) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_weight, input));
  }

  if (output_mask[2]) {
    ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(grad_bias, input));
  }

  ET_CHECK_OR_RETURN_FALSE(
      check_convolution_args(
          input,
          weight,
          std::optional<Tensor>(),
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          grad_output),
      "Invalid convolution arguments");

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_convolution_out_target_size(
      input,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_sizes,
      &output_ndim);

  ET_LOG_AND_RETURN_IF_FALSE(
      output_size_is_valid({output_sizes, output_ndim}, input.dim() - 2));

  ET_CHECK_OR_RETURN_FALSE(
      grad_output.dim() == input.dim(),
      "grad_output should have same number of dimensions as input; grad_output.dim() = %" ET_PRI_TENSOR_DIM
      ", input.dim() = %" ET_PRI_TENSOR_DIM,
      grad_output.dim(),
      input.dim());

  ET_LOG_AND_RETURN_IF_FALSE(
      tensor_has_expected_size(grad_output, {output_sizes, output_ndim}));

  return true;
}

template <typename CTYPE>
void conv2d_backward_impl(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    executorch::aten::ArrayRef<bool> output_mask,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  auto batch_size = input.size(0);
  auto in_channels = input.size(1);
  auto out_channels = weight.size(0);
  auto in_height = input.size(2);
  auto in_width = input.size(3);
  auto out_height = grad_output.size(2);
  auto out_width = grad_output.size(3);
  auto kernel_height = weight.size(2);
  auto kernel_width = weight.size(3);

  const int64_t stride_h = val_at(stride, 0);
  const int64_t padding_h = val_at(padding, 0, /*default_value=*/0);
  const int64_t dilation_h = val_at(dilation, 0);
  const int64_t stride_w = val_at(stride, 1);
  const int64_t padding_w = val_at(padding, 1, /*default_value=*/0);
  const int64_t dilation_w = val_at(dilation, 1);

  auto in_channels_per_group = in_channels / groups;
  auto out_channels_per_group = out_channels / groups;

  const CTYPE* grad_output_data = grad_output.const_data_ptr<CTYPE>();
  const CTYPE* input_data = input.const_data_ptr<CTYPE>();
  const CTYPE* weight_data = weight.const_data_ptr<CTYPE>();

  CTYPE* grad_input_data = nullptr;
  CTYPE* grad_weight_data = nullptr;
  CTYPE* grad_bias_data = nullptr;

  if (output_mask[0]) {
    grad_input_data = grad_input.mutable_data_ptr<CTYPE>();
    memset(grad_input_data, 0, grad_input.nbytes());
  }

  if (output_mask[1]) {
    grad_weight_data = grad_weight.mutable_data_ptr<CTYPE>();
    memset(grad_weight_data, 0, grad_weight.nbytes());
  }

  if (output_mask[2]) {
    grad_bias_data = grad_bias.mutable_data_ptr<CTYPE>();
    memset(grad_bias_data, 0, grad_bias.nbytes());
  }

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType out_coord[kTensorDimensionLimit];
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType in_coord[kTensorDimensionLimit];
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType weight_coord[kTensorDimensionLimit];

  // Compute gradients
  for (int64_t b = 0; b < batch_size; ++b) { // Loop over each batch
    in_coord[0] = b;
    out_coord[0] = b;
    for (int64_t g = 0; g < groups; ++g) { // Loop over each group
      for (int64_t h = 0; h < out_height; ++h) { // Loop over each output row
        out_coord[2] = h;
        for (int64_t w = 0; w < out_width; ++w) { // Loop over each output col
          out_coord[3] = w;

          // Loop over each output channel in the group
          for (int64_t oc = 0; oc < out_channels_per_group; ++oc) {
            int64_t oc_global = oc + g * out_channels_per_group;
            weight_coord[0] = oc_global;
            out_coord[1] = oc_global;

            int64_t out_idx = calculate_linear_index(
                out_coord, grad_output.strides().data(), 4);

            // Accumulate the gradient with respect to the bias if required
            if (output_mask[2]) {
              grad_bias_data[oc_global] += grad_output_data[out_idx];
            }

            // Loop over each input channel in the group
            for (int64_t ic = 0; ic < in_channels_per_group; ++ic) {
              int64_t ic_global = ic + g * in_channels_per_group;
              in_coord[1] = ic_global;
              weight_coord[1] = ic;

              // Loop over each element
              for (int64_t kh = 0; kh < kernel_height; ++kh) {
                int64_t in_h = h * stride_h - padding_h + kh * dilation_h;
                if (in_h >= 0 && in_h < in_height) {
                  in_coord[2] = in_h;
                  weight_coord[2] = kh;

                  for (int64_t kw = 0; kw < kernel_width; ++kw) {
                    int64_t in_w = w * stride_w - padding_w + kw * dilation_w;
                    if (in_w >= 0 && in_w < in_width) {
                      in_coord[3] = in_w;
                      weight_coord[3] = kw;

                      int64_t in_idx = calculate_linear_index(
                          in_coord, input.strides().data(), 4);

                      int64_t weight_idx = calculate_linear_index(
                          weight_coord, weight.strides().data(), 4);

                      // Gradient with respect to the input if required
                      if (output_mask[0]) {
                        grad_input_data[in_idx] +=
                            grad_output_data[out_idx] * weight_data[weight_idx];
                      }
                      // Gradient with respect to the weight if required
                      if (output_mask[1]) {
                        grad_weight_data[weight_idx] +=
                            grad_output_data[out_idx] * input_data[in_idx];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

std::tuple<Tensor&, Tensor&, Tensor&> convolution_backward_out(
    KernelRuntimeContext& ctx,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const OptIntArrayRef bias_sizes_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    executorch::aten::ArrayRef<bool> output_mask,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(
      grad_input, grad_weight, grad_bias);

  ET_KERNEL_CHECK(
      ctx,
      check_convolution_backward_args(
          grad_output,
          input,
          weight,
          bias_sizes_opt,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          output_mask,
          grad_input,
          grad_weight,
          grad_bias),
      InvalidArgument,
      ret_val);

  if (output_mask[0]) {
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_input, input.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  if (output_mask[1]) {
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_weight, weight.sizes()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  if (bias_sizes_opt.has_value() && output_mask[2]) {
    ET_KERNEL_CHECK(
        ctx,
        resize_tensor(grad_bias, bias_sizes_opt.value()) == Error::Ok,
        InvalidArgument,
        ret_val);
  }

  static constexpr auto name = "convolution_backward.out";

  ET_SWITCH_FLOATHBF16_TYPES(input.scalar_type(), ctx, name, CTYPE, [&]() {
    conv2d_backward_impl<CTYPE>(
        grad_output,
        input,
        weight,
        stride,
        padding,
        dilation,
        groups,
        output_mask,
        grad_input,
        grad_weight,
        grad_bias);
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch

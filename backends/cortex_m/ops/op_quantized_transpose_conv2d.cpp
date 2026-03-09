/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {
constexpr int64_t kConvTransposeDim = 4;

bool validate_transpose_conv2d_arguments(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts) {
  if (input.dim() != kConvTransposeDim || weight.dim() != kConvTransposeDim ||
      output.dim() != kConvTransposeDim) {
    ET_LOG(Error, "quantized_transpose_conv2d_out: tensors must be 4-D");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (!is_channels_last_tensor(input)) {
    ET_LOG(
        Error, "quantized_transpose_conv2d_out: input must be channels_last");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (!is_channels_last_tensor(output)) {
    ET_LOG(
        Error, "quantized_transpose_conv2d_out: output must be channels_last");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.scalar_type() != ScalarType::Char ||
      output.scalar_type() != ScalarType::Char) {
    ET_LOG(
        Error, "quantized_transpose_conv2d_out: input and output must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (weight.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "quantized_transpose_conv2d_out: weight must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (bias.has_value() && bias.value().scalar_type() != ScalarType::Int) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: bias must be int32 if provided");
    context.fail(Error::InvalidArgument);
    return false;
  }

  const int64_t out_channels = output.size(1);
  if (requantize_multipliers.size(0) != out_channels ||
      requantize_shifts.size(0) != out_channels) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: per-channel params must match output channels (%zd)",
        out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}
} // namespace

Tensor& quantized_transpose_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef output_padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  if (!validate_transpose_conv2d_arguments(
          context,
          input,
          weight,
          bias,
          out,
          requantize_multipliers,
          requantize_shifts)) {
    return out;
  }

  ET_CHECK_MSG(
      output_padding[0] == 0 && output_padding[1] == 0,
      "quantized_transpose_conv2d: non-zero output_padding is not supported");

  const int32_t batch = static_cast<int32_t>(input.size(0));
  const int32_t input_channels = static_cast<int32_t>(input.size(1));
  const int32_t input_height = static_cast<int32_t>(input.size(2));
  const int32_t input_width = static_cast<int32_t>(input.size(3));

  const int32_t kernel_output_channels = static_cast<int32_t>(weight.size(0));
  const int32_t kernel_height = static_cast<int32_t>(weight.size(1));
  const int32_t kernel_width = static_cast<int32_t>(weight.size(2));
  const int32_t kernel_input_channels = static_cast<int32_t>(weight.size(3));

  const int32_t output_channels = static_cast<int32_t>(out.size(1));
  const int32_t output_height = static_cast<int32_t>(out.size(2));
  const int32_t output_width = static_cast<int32_t>(out.size(3));

  if (kernel_output_channels != output_channels) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: weight output channels (%d) != output channels (%d)",
        kernel_output_channels,
        output_channels);
    context.fail(Error::InvalidArgument);
    return out;
  }

  ET_CHECK_MSG(
      weight.size(3) == input_channels,
      "quantized_transpose_conv2d: weight input channels (%d) must match input channels (%d)",
      static_cast<int>(weight.size(3)),
      static_cast<int>(input_channels));

  const int32_t input_offset_val = static_cast<int32_t>(input_offset);
  const int32_t output_offset_val = static_cast<int32_t>(output_offset);
  const int32_t activation_min_val = static_cast<int32_t>(activation_min);
  const int32_t activation_max_val = static_cast<int32_t>(activation_max);

  const int32_t stride_h = static_cast<int32_t>(stride[0]);
  const int32_t stride_w = static_cast<int32_t>(stride[1]);
  const int32_t pad_h = static_cast<int32_t>(padding[0]);
  const int32_t pad_w = static_cast<int32_t>(padding[1]);
  const int32_t dil_h = static_cast<int32_t>(dilation[0]);
  const int32_t dil_w = static_cast<int32_t>(dilation[1]);

  const int32_t* multiplier_data =
      requantize_multipliers.const_data_ptr<int32_t>();
  const int32_t* shift_data = requantize_shifts.const_data_ptr<int32_t>();

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data = weight.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;

  // Reference transposed conv (output-centric, channels-last NHWC layout).
  // Weight layout: [output_channels, kernel_h, kernel_w, input_channels] (OHWI).
  // For each output position (n, oh, ow, oc), the contributing input positions
  // satisfy: ih = (oh - kh*dil_h + pad_h) / stride_h (must be a non-negative
  // integer within input bounds), and similarly for iw.
  for (int32_t n = 0; n < batch; ++n) {
    for (int32_t oh = 0; oh < output_height; ++oh) {
      for (int32_t ow = 0; ow < output_width; ++ow) {
        for (int32_t oc = 0; oc < output_channels; ++oc) {
          int32_t acc = bias_data != nullptr ? bias_data[oc] : 0;

          for (int32_t kh = 0; kh < kernel_height; ++kh) {
            const int32_t ih_raw = oh - kh * dil_h + pad_h;
            if (ih_raw < 0 || ih_raw % stride_h != 0) {
              continue;
            }
            const int32_t ih = ih_raw / stride_h;
            if (ih >= input_height) {
              continue;
            }

            for (int32_t kw = 0; kw < kernel_width; ++kw) {
              const int32_t iw_raw = ow - kw * dil_w + pad_w;
              if (iw_raw < 0 || iw_raw % stride_w != 0) {
                continue;
              }
              const int32_t iw = iw_raw / stride_w;
              if (iw >= input_width) {
                continue;
              }

              for (int32_t ic = 0; ic < input_channels; ++ic) {
                const int64_t in_idx =
                    ((static_cast<int64_t>(n) * input_height + ih) *
                         input_width +
                     iw) *
                        input_channels +
                    ic;
                const int64_t w_idx =
                    ((static_cast<int64_t>(oc) * kernel_height + kh) *
                         kernel_width +
                     kw) *
                        input_channels +
                    ic;
                acc += (static_cast<int32_t>(input_data[in_idx]) +
                        input_offset_val) *
                       static_cast<int32_t>(weight_data[w_idx]);
              }
            }
          }

          // Per-channel requantization: result = round(acc * multiplier / 2^(31-shift))
          const int32_t mul = multiplier_data[oc];
          const int32_t sft = shift_data[oc];
          const int32_t right_shift = 31 - sft;
          const int64_t acc64 = static_cast<int64_t>(acc) * mul;
          int32_t result;
          if (right_shift > 0) {
            result = static_cast<int32_t>(
                (acc64 + (1LL << (right_shift - 1))) >> right_shift);
          } else {
            result = static_cast<int32_t>(acc64 << (-right_shift));
          }

          result += output_offset_val;
          result = result < activation_min_val ? activation_min_val : result;
          result = result > activation_max_val ? activation_max_val : result;
          const int64_t out_idx =
              ((static_cast<int64_t>(n) * output_height + oh) *
                   output_width +
               ow) *
                  output_channels +
              oc;
          output_data[out_idx] = static_cast<int8_t>(result);
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace cortex_m

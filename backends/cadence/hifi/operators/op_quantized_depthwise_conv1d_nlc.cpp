/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_nlc.h>
#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

using Tensor = executorch::aten::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using ScalarType = executorch::aten::ScalarType;
using ::executorch::aten::IntArrayRef;

namespace impl {
namespace HiFi {
namespace native {

namespace {

// Optimized depthwise conv1d NLC using NNLib's conv2d depthwise kernel
// with kernel_height=1. Handles both int8 and uint8 via the same
// xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s function (uint8 is cast).
//
// Input:  [N, L, C] (NLC format)
// Weight: [OC, K, 1] (NLC depthwise, IC/groups == 1)
// Output: [N, OL, OC]
//
// NNLib expects depthwise weight in [K, OC] format, so we transpose
// the weight from [OC, K] (squeezed from [OC, K, 1]) to [K, OC].
void xa_opt_quantized_depthwise_conv1d_nlc(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    int32_t in_zero_point,
    int32_t weight_zero_point,
    float bias_scale,
    float output_scale,
    int32_t output_zero_point,
    Tensor& out) {
  WORD8* __restrict__ p_out =
      (WORD8* __restrict__)out.mutable_data_ptr<int8_t>();
  WORD8* __restrict__ p_inp =
      (WORD8* __restrict__)input.const_data_ptr<int8_t>();
  WORD8* __restrict__ p_kernel =
      (WORD8* __restrict__)weight.const_data_ptr<int8_t>();
  WORD32* __restrict__ p_bias =
      (WORD32* __restrict__)bias.const_data_ptr<int32_t>();

  // NLC format: [N, L, C]
  WORD32 batches = input.size(0);
  WORD32 input_width = input.size(1);
  WORD32 input_channels = input.size(2);
  WORD32 input_height = 1;

  // Weight: [OC, K, IC/groups] where IC/groups == 1 for depthwise
  WORD32 out_channels = weight.size(0);
  WORD32 kernel_width = weight.size(1);
  WORD32 kernel_height = 1;

  WORD32 out_width = out.size(1);
  WORD32 out_height = 1;

  // For 1D conv: stride/padding are 1-element arrays
  WORD32 x_stride = stride[stride.size() - 1];
  WORD32 y_stride = 1;
  WORD32 x_padding = padding[padding.size() - 1];
  WORD32 y_padding = 0;

  WORD32 input_zero_bias = -in_zero_point;
  WORD32 out_zero_bias = output_zero_point;
  WORD32 inp_precision = 8;

  WORD32 channels_multiplier = out_channels / input_channels;

  // Per-channel output multiplier/shift (uniform for per-tensor quantization)
  WORD32 out_multiplier32[out_channels];
  WORD32 out_shift32[out_channels];

  float out_scale = 1. / output_scale;

  for (int i = 0; i < out_channels; i++) {
    out_multiplier32[i] = bias_scale * out_scale * 2147483648;
    out_shift32[i] = 0;
  }

  // Transpose weight from [OC, K, 1] (effectively [OC, K]) to [K, OC]
  // which is the format NNLib depthwise expects.
  constexpr int kNnlibMaxDim = 5;

  WORD8* ptr_weight = (WORD8*)kernels::allocate_temp_memory(
      ctx, ((out_channels * kernel_width) + 8) * sizeof(WORD8));
  WORD8* p_transposed_kernel = (WORD8*)ALIGN_PTR(ptr_weight, 8);

  WORD32 p_kernel_shape[kNnlibMaxDim] = {1, 1, 1, out_channels, kernel_width};
  WORD32 p_kernel_out_shape[kNnlibMaxDim] = {
      1, 1, 1, kernel_width, out_channels};
  WORD32 p_permute_vec[kNnlibMaxDim] = {0, 1, 2, 4, 3};

  xa_nn_transpose_8_8(
      p_transposed_kernel,
      p_kernel_out_shape,
      p_kernel,
      p_kernel_shape,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);

  // Get scratch buffer for depthwise conv
  WORD32 scratch_size = xa_nn_conv2d_depthwise_getsize(
      input_height,
      input_width,
      input_channels,
      kernel_height,
      kernel_width,
      channels_multiplier,
      x_stride,
      y_stride,
      x_padding,
      y_padding,
      out_height,
      out_width,
      inp_precision,
      0); // NHWC

  scratch_size = scratch_size < 0 ? 0 : scratch_size;

  WORD32* ptr_scratch =
      (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);
  pVOID p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

  for (int _n = 0; _n < batches; _n++) {
    WORD8* in_batch = p_inp + _n * input_channels * input_height * input_width;
    WORD8* out_batch = p_out + _n * out_channels * out_height * out_width;

    xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
        out_batch,
        p_transposed_kernel,
        in_batch,
        p_bias,
        input_height,
        input_width,
        input_channels,
        kernel_height,
        kernel_width,
        channels_multiplier,
        x_stride,
        y_stride,
        x_padding,
        y_padding,
        out_height,
        out_width,
        input_zero_bias,
        out_multiplier32,
        out_shift32,
        out_zero_bias,
        0, // inp_data_format = NHWC
        0, // out_data_format = NHWC
        p_scratch);
  }
}

} // namespace

void quantized_depthwise_conv1d_nlc_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    Tensor& out) {
  // Fall back to generic for dilation != 1, since NNLib depthwise
  // does not support dilation.
  if (dilation[dilation.size() - 1] != 1) {
    impl::generic::native::quantized_conv1d_nlc_per_tensor_out(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out_multiplier,
        out_shift,
        out);
    return;
  }

  ScalarType dtype = out.scalar_type();

  if (dtype == ScalarType::Char || dtype == ScalarType::Byte) {
    // Both int8 and uint8 use the same NNLib function
    // (uint8 is cast to int8 internally by NNLib)
    xa_opt_quantized_depthwise_conv1d_nlc(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        static_cast<int32_t>(in_zero_point),
        static_cast<int32_t>(weight_zero_point),
        static_cast<float>(bias_scale),
        static_cast<float>(output_scale),
        static_cast<int32_t>(output_zero_point),
        out);
  } else {
    ET_DCHECK_MSG(
        false,
        "Unhandled dtype %s for quantized_depthwise_conv1d_nlc",
        torch::executor::toString(dtype));
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl

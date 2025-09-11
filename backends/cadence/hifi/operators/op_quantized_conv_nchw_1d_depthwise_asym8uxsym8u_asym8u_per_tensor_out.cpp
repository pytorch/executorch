/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/backends/cadence/hifi/operators/operators.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#define ALIGN_PTR(x, bytes) ((((unsigned)(x)) + (bytes - 1)) & (~(bytes - 1)))

using Tensor = executorch::aten::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using ScalarType = executorch::aten::ScalarType;
using ::executorch::aten::IntArrayRef;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

// Specialized 1D depthwise NCHW convolution for uint8 x uint8 -> uint8
void xa_opt_quantized_conv_nchw_1d_depthwise_asym8uxsym8u_asym8u(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    int32_t in_zero_point,
    int32_t weight_zero_point,
    float bias_scale,
    float output_scale,
    int32_t output_zero_point,
    Tensor& out) {
  constexpr int kNnlibMaxDim = 4;

  UWORD8* __restrict__ p_out =
      (UWORD8* __restrict__)out.mutable_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_inp =
      (UWORD8* __restrict__)input.const_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_kernel =
      (UWORD8* __restrict__)weight.const_data_ptr<uint8_t>();
  WORD32* __restrict__ p_bias =
      (WORD32* __restrict__)bias.const_data_ptr<int32_t>();

  // For 1D convolution, height is always 1
  WORD32 input_height = 1;
  WORD32 input_width = input.size(2);
  WORD32 input_channels = input.size(1);
  WORD32 kernel_height = 1;
  WORD32 kernel_width = weight.size(2);
  WORD32 out_channels = weight.size(0);
  WORD32 out_height = 1;
  WORD32 out_width = out.size(2);
  WORD32 batches = input.size(0);

  WORD32 x_stride = stride[1];
  WORD32 y_stride = stride[0];
  WORD32 x_padding = padding[1];
  WORD32 y_padding = padding[0];

  WORD32 input_zero_bias = in_zero_point;
  WORD32 out_zero_bias = output_zero_point;
  WORD32 inp_precision = 8;

  WORD32 channels_multiplier = out_channels / input_channels;

  WORD32 out_multiplier32[out_channels];
  WORD32 out_shift32[out_channels];

  float out_scale = 1. / output_scale;

  for (int i = 0; i < out_channels; i++) {
    out_multiplier32[i] = bias_scale * out_scale * 2147483648;
    out_shift32[i] = 0;
  }

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
      1); // NCHW

  scratch_size = scratch_size < 0 ? 0 : scratch_size;

  WORD32* ptr_scratch =
      (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);
  pVOID p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

  UWORD8* ptr1 = (UWORD8*)kernels::allocate_temp_memory(
      ctx,
      ((batches * out_channels * out_height * out_width) + 8) * sizeof(UWORD8));

  UWORD8* p_out_temp = (UWORD8*)ALIGN_PTR(ptr1, 8);

  for (int _n = 0; _n < batches; _n++) {
    UWORD8* in_batch = p_inp + _n * input_channels * input_height * input_width;
    UWORD8* out_batch = p_out_temp + _n * out_channels * out_height * out_width;

    xa_nn_conv2d_depthwise_per_chan_sym8sxasym8s(
        (WORD8*)out_batch,
        (WORD8*)p_kernel,
        (WORD8*)in_batch,
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
        1, // NCHW
        0, // NHWC
        p_scratch);
  }

  WORD32 p_inp_shape[kNnlibMaxDim];
  p_inp_shape[0] = batches;
  p_inp_shape[1] = out_height;
  p_inp_shape[2] = out_width;
  p_inp_shape[3] = out_channels;

  WORD32 p_out_shape[kNnlibMaxDim];
  p_out_shape[0] = batches;
  p_out_shape[1] = out_channels;
  p_out_shape[2] = out_height;
  p_out_shape[3] = out_width;

  WORD32 p_permute_vec[kNnlibMaxDim] = {0, 3, 1, 2};

  xa_nn_transpose_8_8(
      (WORD8*)p_out,
      p_out_shape,
      (WORD8*)p_out_temp,
      p_inp_shape,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);
}

void quantized_conv_nchw_1d_depthwise_asym8uxsym8u_asym8u_per_tensor_out(
    __ET_UNUSED KernelRuntimeContext& ctx,
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
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  xa_opt_quantized_conv_nchw_1d_depthwise_asym8uxsym8u_asym8u(
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
      out);
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence

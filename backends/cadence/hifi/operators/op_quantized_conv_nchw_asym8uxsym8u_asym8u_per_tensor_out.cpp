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

namespace impl {
namespace HiFi {
namespace native {

// Optimized NCHW convolution for uint8 x uint8 -> uint8
void xa_opt_quantized_conv_nchw_asym8uxsym8u_asym8u(
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
  bool conv1d = input.dim() == 3;
  constexpr int kNnlibMaxDim = 4;

  UWORD8* __restrict__ p_out =
      (UWORD8* __restrict__)out.mutable_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_inp =
      (UWORD8* __restrict__)input.const_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_kernel =
      (UWORD8* __restrict__)weight.const_data_ptr<uint8_t>();
  WORD32* __restrict__ p_bias =
      (WORD32* __restrict__)bias.const_data_ptr<int32_t>();

  WORD32 input_height = conv1d ? 1 : input.size(2);
  WORD32 input_width = conv1d ? input.size(2) : input.size(3);
  WORD32 input_channels = input.size(1);
  WORD32 kernel_height = conv1d ? 1 : weight.size(2);
  WORD32 kernel_width = conv1d ? weight.size(2) : weight.size(3);
  WORD32 kernel_channels = weight.size(1);
  WORD32 out_channels = weight.size(0);
  WORD32 out_height = conv1d ? 1 : out.size(2);
  WORD32 out_width = conv1d ? out.size(2) : out.size(3);
  WORD32 batches = input.size(0);

  WORD32 x_stride = stride[1];
  WORD32 y_stride = stride[0];
  WORD32 x_padding = padding[1];
  WORD32 y_padding = padding[0];
  WORD32 dilation_width = dilation[1];
  WORD32 dilation_height = dilation[0];

  WORD32 input_zero_bias = -in_zero_point;
  WORD32 kernel_zero_bias = -weight_zero_point;

  WORD32 out_multiplier32[out_channels];
  WORD32 out_shift32[out_channels];

  float out_scale = 1. / output_scale;

  for (int i = 0; i < out_channels; i++) {
    out_multiplier32[i] = bias_scale * out_scale * 2147483648;
    out_shift32[i] = 0;
  }

  WORD32 out_zero_bias = output_zero_point;
  WORD32 inp_precision = 8;
  WORD32 kernel_precision = 8;
  pVOID p_scratch = nullptr;
  WORD32* ptr_scratch;

  WORD32 scratch_size = 0;

  ET_CHECK_MSG(groups == 1, "Only groups=1 supported for regular convolution");
  WORD32 out_data_format = 1;

  UWORD8* ptr1 = (UWORD8*)kernels::allocate_temp_memory(
      ctx,
      ((batches * input_channels * input_height * input_width) + 8) *
          sizeof(UWORD8));

  UWORD8* ptr2 = (UWORD8*)kernels::allocate_temp_memory(
      ctx,
      ((out_channels * kernel_channels * kernel_height * kernel_width) + 8) *
          sizeof(UWORD8));

  UWORD8* pin = (UWORD8*)ALIGN_PTR(ptr1, 8);
  UWORD8* pkernel = (UWORD8*)ALIGN_PTR(ptr2, 8);

  WORD32 p_inp_shape[kNnlibMaxDim];
  p_inp_shape[0] = input.size(0);
  p_inp_shape[1] = input_channels;
  p_inp_shape[2] = input_height;
  p_inp_shape[3] = input_width;

  WORD32 p_out_shape[kNnlibMaxDim];
  p_out_shape[0] = input.size(0);
  p_out_shape[1] = input_height;
  p_out_shape[2] = input_width;
  p_out_shape[3] = input_channels;

  WORD32 p_permute_vec[kNnlibMaxDim] = {0, 2, 3, 1};

  xa_nn_transpose_8_8(
      (WORD8*)pin,
      p_out_shape,
      (WORD8*)p_inp,
      p_inp_shape,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);

  WORD32 p_inp_shape1[kNnlibMaxDim];
  p_inp_shape1[0] = out_channels;
  p_inp_shape1[1] = kernel_channels;
  p_inp_shape1[2] = kernel_height;
  p_inp_shape1[3] = kernel_width;

  WORD32 p_out_shape1[kNnlibMaxDim];
  p_out_shape1[0] = out_channels;
  p_out_shape1[1] = kernel_height;
  p_out_shape1[2] = kernel_width;
  p_out_shape1[3] = kernel_channels;

  xa_nn_transpose_8_8(
      (WORD8*)pkernel,
      p_out_shape1,
      (WORD8*)p_kernel,
      p_inp_shape1,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);

  scratch_size = xa_nn_conv2d_getsize(
      input_height,
      input_width,
      input_channels,
      kernel_height,
      kernel_width,
      kernel_channels,
      dilation_height,
      dilation_width,
      y_stride,
      y_padding,
      x_stride,
      x_padding,
      out_height,
      out_width,
      out_channels,
      inp_precision,
      kernel_precision,
      out_data_format);

  scratch_size = scratch_size < 0 ? 0 : scratch_size;

  ptr_scratch = (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);

  p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

  for (int _n = 0; _n < batches; _n++) {
    UWORD8* in_batch = pin + _n * input_channels * input_height * input_width;
    UWORD8* out_batch = p_out + _n * out_channels * out_height * out_width;

    xa_nn_conv2d_per_chan_sym8sxasym8s(
        (WORD8*)out_batch,
        (WORD8*)in_batch,
        (WORD8*)pkernel,
        p_bias,
        input_height,
        input_width,
        input_channels,
        kernel_height,
        kernel_width,
        kernel_channels,
        dilation_height,
        dilation_width,
        out_channels,
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
        out_data_format,
        p_scratch);
  }
}

void quantized_conv_nchw_asym8uxsym8u_asym8u_per_tensor_out(
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
  xa_opt_quantized_conv_nchw_asym8uxsym8u_asym8u(
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

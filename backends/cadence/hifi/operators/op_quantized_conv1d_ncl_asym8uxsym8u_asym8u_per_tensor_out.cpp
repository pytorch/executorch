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

// Optimized NCHW 1D convolution for uint8 x uint8 -> uint8
void xa_opt_quantized_conv1d_ncl_asym8uxsym8u_asym8u(
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
  constexpr int kNnlibMaxDim = 5;

  UWORD8* __restrict__ p_out =
      (UWORD8* __restrict__)out.mutable_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_inp =
      (UWORD8* __restrict__)input.const_data_ptr<uint8_t>();
  UWORD8* __restrict__ p_kernel =
      (UWORD8* __restrict__)weight.const_data_ptr<uint8_t>();
  WORD32* __restrict__ p_bias =
      (WORD32* __restrict__)bias.const_data_ptr<int32_t>();

  WORD32 batches = input.size(0);
  WORD32 input_channels = input.size(1);
  WORD32 input_width = input.size(2);
  WORD32 input_height = 1;
  WORD32 kernel_height = 1;
  WORD32 out_channels = weight.size(0);
  WORD32 kernel_channels = weight.size(1);
  WORD32 kernel_width = weight.size(2);
  WORD32 out_width = out.size(2);
  WORD32 out_height = 1;
  WORD32 x_stride = stride[1];
  WORD32 x_padding = padding[1];
  WORD32 input_zero_bias = -in_zero_point;
  WORD32 out_multiplier32 = bias_scale * (1. / output_scale) * 2147483648;
  WORD32 out_shift32 = 0;
  WORD32 kernel_zero_bias = -weight_zero_point;

  WORD32 out_zero_bias = output_zero_point;
  WORD32 out_data_format = 1;

  WORD32 scratch_size =
      xa_nn_conv1d_std_getsize(kernel_width, input_width, input_channels, 8);
  scratch_size = scratch_size < 0 ? 0 : scratch_size;
  WORD32* ptr_scratch =
      (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);
  pVOID p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

  WORD8* ptr1 = (WORD8*)kernels::allocate_temp_memory(
      ctx, ((batches * input_channels * input_width) + 8) * sizeof(WORD8));
  WORD8* ptr2 = (WORD8*)kernels::allocate_temp_memory(
      ctx,
      ((out_channels * kernel_channels * kernel_width) + 8) * sizeof(WORD8));
  WORD8* pin = (WORD8*)ALIGN_PTR(ptr1, 8);
  WORD8* pkernel = (WORD8*)ALIGN_PTR(ptr2, 8);

  WORD32 p_inp_shape[kNnlibMaxDim];
  p_inp_shape[0] = 1;
  p_inp_shape[1] = 1;
  p_inp_shape[2] = batches;
  p_inp_shape[3] = input_channels;
  p_inp_shape[4] = input_width;

  WORD32 p_out_shape[kNnlibMaxDim];
  p_out_shape[0] = 1;
  p_out_shape[1] = 1;
  p_out_shape[2] = batches;
  p_out_shape[3] = input_width;
  p_out_shape[4] = input_channels;

  WORD32 p_permute_vec[kNnlibMaxDim] = {0, 1, 2, 4, 3};

  xa_nn_transpose_8_8(
      (WORD8*)pin,
      p_out_shape,
      (WORD8*)p_inp,
      p_inp_shape,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);

  WORD32 p_inp_shape1[kNnlibMaxDim];
  p_inp_shape1[0] = 1;
  p_inp_shape1[1] = 1;
  p_inp_shape1[2] = out_channels;
  p_inp_shape1[3] = kernel_channels;
  p_inp_shape1[4] = kernel_width;

  WORD32 p_out_shape1[kNnlibMaxDim];
  p_out_shape1[0] = 1;
  p_out_shape1[1] = 1;
  p_out_shape1[2] = out_channels;
  p_out_shape1[3] = kernel_width;
  p_out_shape1[4] = kernel_channels;

  xa_nn_transpose_8_8(
      (WORD8*)pkernel,
      p_out_shape1,
      (WORD8*)p_kernel,
      p_inp_shape1,
      p_permute_vec,
      kNnlibMaxDim,
      kNnlibMaxDim);

  for (int _n = 0; _n < batches; _n++) {
    UWORD8* in_batch = (UWORD8*)(pin + _n * input_channels * input_width);
    UWORD8* out_batch = (UWORD8*)(p_out + _n * out_channels * out_width);

    xa_nn_conv1d_std_asym8uxasym8u(
        out_batch,
        in_batch,
        (UWORD8*)pkernel,
        p_bias,
        input_width,
        input_height,
        input_channels,
        kernel_width,
        out_channels,
        x_stride,
        x_padding,
        out_width,
        input_zero_bias,
        kernel_zero_bias,
        out_multiplier32,
        out_shift32,
        out_zero_bias,
        out_data_format,
        p_scratch);
  }
}

void quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    __ET_UNUSED IntArrayRef dilation,
    __ET_UNUSED int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  xa_opt_quantized_conv1d_ncl_asym8uxsym8u_asym8u(
      ctx,
      input,
      weight,
      bias,
      stride,
      padding,
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

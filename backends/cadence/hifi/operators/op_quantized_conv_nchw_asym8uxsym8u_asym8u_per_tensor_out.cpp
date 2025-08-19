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

// This implements a specialized uint8 x uint8 -> uint8 quantized 2d conv kernel
// for NCHW layout. This variant is optimized for asymmetric uint8 inputs,
// weights, and outputs. The input is of shape [n x c x h x w] The weight is of
// shape [oc x wc x wh x ww], where wc == c The output is of shape [n x oc x oh
// x ow] The bias is of shape [oc]
template <bool quantized = true>
__attribute__((noinline)) void conv2d_nchw_asym8uxsym8u_asym8u_core(
    // All the arrays
    const uint8_t* __restrict__ p_in,
    const uint8_t* __restrict__ p_weight,
    const int32_t* __restrict__ p_bias,
    uint8_t* __restrict__ p_out,
    // The array sizes
    int32_t n,
    int32_t c,
    int32_t h,
    int32_t w,
    int32_t oc,
    int32_t wc,
    int32_t wh,
    int32_t ww,
    int32_t oh,
    int32_t ow,
    // Stride
    int16_t s0,
    int16_t s1,
    // Padding
    int16_t p0,
    int16_t p1,
    // Dilation
    int16_t d0,
    int16_t d1,
    // Group for depthwise conv
    int16_t groups,
    // Quantization parameters
    uint8_t in_zero_point = 0,
    int32_t weight_zero_point = 0,
    float bias_scale = 1,
    float out_scale = 1,
    uint8_t out_zero_point = 0) {
  float inv_out_scale = 1. / out_scale;
  bool zero_pad_unit_dilation = d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0;

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const uint8_t* in_batch = p_in + _n * c * h * w;
    uint8_t* out_batch = p_out + _n * oc * oh * ow;
    // Compute separable convolution for each group
    for (int _g = 0; _g < groups; ++_g) {
      // Identify the input and output channels involved in the computation
      // of this group
      int sic = _g * icpg;
      int soc = _g * ocpg;
      // Populate all the output channels in the group
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        uint8_t* out_plane = out_batch + _oc * oh * ow;
        const uint8_t* weight_batch = p_weight + _oc * wc * wh * ww;
        // We compute one output channel at a time. The computation can be
        // thought of as a stencil computation: we iterate over an input of size
        // icpg x h x w, with a stencil of size icpg x wh x ww, to compute an
        // output channel of size 1 x oh x ow.
        for (int _h = 0, _oh = 0; _oh < oh; _h += s0, ++_oh) {
          for (int _w = 0, _ow = 0; _ow < ow; _w += s1, ++_ow) {
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to the
            // output channel being computed) with the corresponding weight
            // channel.
            // Optimized path for zero padding and unit dilation
            if (zero_pad_unit_dilation) {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const uint8_t* in_plane = in_batch + _ic * h * w;
                const uint8_t* weight_plane =
                    weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int ioff = (_h + _wh) * w + (_w + _ww);
                    int woff = _wh * ww + _ww;
                    float lhs = static_cast<float>(in_plane[ioff]) -
                        static_cast<float>(in_zero_point);
                    float rhs = static_cast<float>(weight_plane[woff]) -
                        static_cast<float>(weight_zero_point);
                    acc += lhs * rhs;
                  }
                }
              }
            } else {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const uint8_t* in_plane = in_batch + _ic * h * w;
                const uint8_t* weight_plane =
                    weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    if (((_h + d0 * _wh - p0) >= 0) &&
                        ((_h + d0 * _wh - p0) < h) &&
                        ((_w + d1 * _ww - p1) >= 0) &&
                        ((_w + d1 * _ww - p1) < w)) {
                      int ioff =
                          (_h + d0 * _wh - p0) * w + (_w + d1 * _ww - p1);
                      int woff = _wh * ww + _ww;
                      float lhs = static_cast<float>(in_plane[ioff]) -
                          static_cast<float>(in_zero_point);
                      float rhs = static_cast<float>(weight_plane[woff]) -
                          static_cast<float>(weight_zero_point);
                      acc += lhs * rhs;
                    }
                  }
                }
              }
            }
            // Quantize the accumulated result
            float val = bias_scale * acc;
            out_plane[_oh * ow + _ow] =
                kernels::quantize<uint8_t>(val, inv_out_scale, out_zero_point);
          }
        }
      }
    }
  }
}

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

  if (groups == 1) {
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
    return;
  }

  if (groups == input_channels) {
    WORD32 channels_multiplier = out_channels / input_channels;

    scratch_size = xa_nn_conv2d_depthwise_getsize(
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

    ptr_scratch = (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);

    p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

    UWORD8* ptr1 = (UWORD8*)kernels::allocate_temp_memory(
        ctx,
        ((batches * out_channels * out_height * out_width) + 8) *
            sizeof(UWORD8));

    UWORD8* p_out_temp = (UWORD8*)ALIGN_PTR(ptr1, 8);

    for (int _n = 0; _n < batches; _n++) {
      UWORD8* in_batch =
          p_inp + _n * input_channels * input_height * input_width;
      UWORD8* out_batch =
          p_out_temp + _n * out_channels * out_height * out_width;

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

    return;
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
  bool optimized = true;

  // Disable optimization for dilated convolutions
  if ((dilation[0] != 1) || (dilation[1] != 1))
    optimized = false;

  if (optimized) {
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
  } else {
    bool conv1d = input.dim() == 3;
    // input = [n, c, h, w]
    const int n = input.size(0);
    const int c = input.size(1);
    const int h = conv1d ? 1 : input.size(2);
    const int w = conv1d ? input.size(2) : input.size(3);
    // weight = [oc, wc, wh, ww]
    const int oc = weight.size(0);
    const int wc = weight.size(1);
    const int wh = conv1d ? 1 : weight.size(2);
    const int ww = conv1d ? weight.size(2) : weight.size(3);
    // output = [n, oc, oh, ow]
    const int oh = conv1d ? 1 : out.size(2);
    const int ow = conv1d ? out.size(2) : out.size(3);

    conv2d_nchw_asym8uxsym8u_asym8u_core(
        input.const_data_ptr<uint8_t>(),
        weight.const_data_ptr<uint8_t>(),
        bias.const_data_ptr<int32_t>(),
        out.mutable_data_ptr<uint8_t>(),
        n,
        c,
        h,
        w,
        oc,
        wc,
        wh,
        ww,
        oh,
        ow,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        static_cast<uint8_t>(in_zero_point),
        weight_zero_point,
        bias_scale,
        output_scale,
        static_cast<uint8_t>(output_zero_point));
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence

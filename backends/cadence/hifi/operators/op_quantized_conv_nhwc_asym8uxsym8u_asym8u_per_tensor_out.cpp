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
// for NHWC layout. This variant is optimized for asymmetric uint8 inputs,
// weights, and outputs. The input is of shape [n x h x w x c] The weight is of
// shape [oc x wh x ww x wc] The output is of shape [n x oh x ow x oc] The bias
// is of shape [oc]
template <bool quantized = true>
__attribute__((noinline)) void conv2d_nhwc_asym8uxsym8u_asym8u_core(
    // All the arrays
    const uint8_t* __restrict__ p_in,
    const uint8_t* __restrict__ p_weight,
    const int32_t* __restrict__ p_bias,
    uint8_t* __restrict__ p_out,
    // The array sizes
    int32_t n,
    int32_t h,
    int32_t w,
    int32_t c,
    int32_t oc,
    int32_t wh,
    int32_t ww,
    int32_t wc,
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
    const uint8_t* in_batch = p_in + _n * h * w * c;
    uint8_t* out_batch = p_out + _n * oh * ow * oc;
    for (int _h = 0, _oh = 0; _oh < oh; _h += s0, ++_oh) {
      for (int _w = 0, _ow = 0; _ow < ow; _w += s1, ++_ow) {
        uint8_t* out_line = out_batch + (_oh * ow + _ow) * oc;
        // Compute separable convolution for each group
        for (int _g = 0; _g < groups; ++_g) {
          // Identify the input and output channels involved in the computation
          // of this group
          int sic = _g * icpg;
          int soc = _g * ocpg;
          // Populate all the output channels in the group
          for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
            const uint8_t* weight_batch = p_weight + _oc * wh * ww * wc;
            // We compute one output channel at a time. The computation can be
            // thought of as a stencil computation: we iterate over an input of
            // size h x w x icpg, with a stencil of size wh x ww x icpg, to
            // compute an output channel of size oh x ow x 1.
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to
            // the output channel being computed) with the corresponding
            // weight channel. Optimized path for zero padding and unit dilation
            if (zero_pad_unit_dilation) {
              for (int _wh = 0; _wh < wh; ++_wh) {
                for (int _ww = 0; _ww < ww; ++_ww) {
                  const uint8_t* in_line =
                      in_batch + (_h + _wh) * w * c + (_w + _ww) * c;
                  const uint8_t* weight_line =
                      weight_batch + _wh * ww * wc + _ww * wc;
                  for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                    float lhs = static_cast<float>(in_line[_ic]) -
                        static_cast<float>(in_zero_point);
                    float rhs = static_cast<float>(weight_line[_ic - sic]) -
                        static_cast<float>(weight_zero_point);
                    acc += lhs * rhs;
                  }
                }
              }
            } else {
              for (int _wh = 0; _wh < wh; ++_wh) {
                for (int _ww = 0; _ww < ww; ++_ww) {
                  if (((_h + d0 * _wh - p0) >= 0) &&
                      ((_h + d0 * _wh - p0) < h) &&
                      ((_w + d1 * _ww - p1) >= 0) &&
                      ((_w + d1 * _ww - p1) < w)) {
                    const uint8_t* in_line = in_batch +
                        (_h + d0 * _wh - p0) * w * c + (_w + d1 * _ww - p1) * c;
                    const uint8_t* weight_line =
                        weight_batch + _wh * ww * wc + _ww * wc;
                    for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                      float lhs = static_cast<float>(in_line[_ic]) -
                          static_cast<float>(in_zero_point);
                      float rhs = static_cast<float>(weight_line[_ic - sic]) -
                          static_cast<float>(weight_zero_point);
                      acc += lhs * rhs;
                    }
                  }
                }
              }
            }
            // Quantize the accumulated result
            float val = bias_scale * acc;
            out_line[_oc] =
                kernels::quantize<uint8_t>(val, inv_out_scale, out_zero_point);
          }
        }
      }
    }
  }
}

// Optimized NHWC convolution for uint8 x uint8 -> uint8
void xa_opt_quantized_conv_nhwc_asym8uxsym8u_asym8u(
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
      UWORD8* in_batch =
          p_inp + _n * input_channels * input_height * input_width;
      UWORD8* out_batch = p_out + _n * out_channels * out_height * out_width;

      xa_nn_conv2d_per_chan_sym8sxasym8s(
          (WORD8*)out_batch,
          (WORD8*)in_batch,
          (WORD8*)p_kernel,
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
        0); // NHWC

    scratch_size = scratch_size < 0 ? 0 : scratch_size;

    ptr_scratch = (WORD32*)kernels::allocate_temp_memory(ctx, scratch_size);

    p_scratch = (pVOID)ALIGN_PTR(ptr_scratch, 8);

    for (int _n = 0; _n < batches; _n++) {
      UWORD8* in_batch =
          p_inp + _n * input_channels * input_height * input_width;
      UWORD8* out_batch = p_out + _n * out_channels * out_height * out_width;

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
          0, // NHWC
          0, // NHWC
          p_scratch);
    }
    return;
  }
}

void quantized_conv_nhwc_asym8uxsym8u_asym8u_per_tensor_out(
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
    xa_opt_quantized_conv_nhwc_asym8uxsym8u_asym8u(
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
    // input = [n, h, w, c]
    const int n = input.size(0);
    const int h = conv1d ? 1 : input.size(1);
    const int w = conv1d ? input.size(1) : input.size(2);
    const int c = conv1d ? input.size(2) : input.size(3);
    // weight = [oc, wh, ww, wc]
    const int oc = weight.size(0);
    const int wh = conv1d ? 1 : weight.size(1);
    const int ww = conv1d ? weight.size(1) : weight.size(2);
    const int wc = conv1d ? weight.size(2) : weight.size(3);
    // output = [n, oh, ow, oc]
    const int oh = conv1d ? 1 : out.size(1);
    const int ow = conv1d ? out.size(1) : out.size(2);

    // Use specialized uint8 kernel
    conv2d_nhwc_asym8uxsym8u_asym8u_core(
        input.const_data_ptr<uint8_t>(),
        weight.const_data_ptr<uint8_t>(),
        bias.const_data_ptr<int32_t>(),
        out.mutable_data_ptr<uint8_t>(),
        n,
        h,
        w,
        c,
        oc,
        wh,
        ww,
        wc,
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

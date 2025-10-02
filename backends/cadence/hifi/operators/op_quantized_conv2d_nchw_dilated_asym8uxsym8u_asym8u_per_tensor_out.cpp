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

using Tensor = executorch::aten::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using ScalarType = executorch::aten::ScalarType;
using ::executorch::aten::IntArrayRef;

namespace impl {
namespace HiFi {
namespace native {

// Dilated fallback implementation for uint8 x uint8 -> uint8 quantized 2d conv
// kernel for NCHW layout. This variant is optimized for asymmetric uint8
// inputs, weights, and outputs. The input is of shape [n x c x h x w] The
// weight is of shape [oc x wc x wh x ww], where wc == c The output is of shape
// [n x oc x oh x ow] The bias is of shape [oc]
template <bool quantized = true>
__attribute__((noinline)) void conv2d_nchw_dilated_asym8uxsym8u_asym8u_core(
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
            // General path for dilated convolutions with padding support
            for (int _ic = sic; _ic < sic + icpg; ++_ic) {
              const uint8_t* in_plane = in_batch + _ic * h * w;
              const uint8_t* weight_plane =
                  weight_batch + (_ic - sic) * wh * ww;
              for (int _wh = 0; _wh < wh; ++_wh) {
                for (int _ww = 0; _ww < ww; ++_ww) {
                  int input_h = _h + d0 * _wh - p0;
                  int input_w = _w + d1 * _ww - p1;
                  if ((input_h >= 0) && (input_h < h) && (input_w >= 0) &&
                      (input_w < w)) {
                    int ioff = input_h * w + input_w;
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

void quantized_conv2d_nchw_dilated_asym8uxsym8u_asym8u_per_tensor_out(
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

  conv2d_nchw_dilated_asym8uxsym8u_asym8u_core(
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

} // namespace native
} // namespace HiFi
} // namespace impl

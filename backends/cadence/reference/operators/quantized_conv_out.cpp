/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>

#include <executorch/runtime/kernel/kernel_includes.h>
#include <algorithm>
#include <cmath>

namespace impl {
namespace reference {
namespace native {

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// This implements a generic 2d conv kernel that operates on raw pointers.
// The version handles both quantized and fp32 convolutions.
// The input is of shape [n x c x h x w]
// The weight is of shape [oc x wc x wh x ww], where wc == c
// The output is of shape [n x oc x oh x ow]
// The bias is of shape [oc]
template <typename IT, typename WT, typename BT, typename OT, bool quantized>
__attribute__((noinline)) void conv2d_nchw_core_generic(
    // All the arrays
    const IT* __restrict__ p_in,
    const WT* __restrict__ p_weight,
    const BT* __restrict__ p_bias,
    OT* __restrict__ p_out,
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
    // Optional args that are only relevant for quantized convolution
    // input zero point
    IT in_zero_point = 0,
    // weight zero point
    const int32_t* __restrict__ weight_zero_point = nullptr,
    const float* __restrict__ bias_scale = nullptr,
    float out_scale = 1,
    OT out_zero_point = 0,
    bool per_tensor_quantized = true) {
  float inv_out_scale = 1. / out_scale;
  bool zero_pad_unit_dilation = d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0;

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * c * h * w;
    OT* out_batch = p_out + _n * oc * oh * ow;
    // Compute separable convolution for each group
    for (int _g = 0; _g < groups; ++_g) {
      // Identify the input and output channels involved in the computation
      // of this group
      int sic = _g * icpg;
      int soc = _g * ocpg;
      // Populate all the output channels in the group
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        OT* out_plane = out_batch + _oc * oh * ow;
        const WT* weight_batch = p_weight + _oc * wc * wh * ww;
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
            // If the padding is 0, and dilation is 1, then we can remove the
            // unnecessary checks, and simplify the code so that it can be
            // vectorized by Tensilica compiler.
            if (zero_pad_unit_dilation) {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int ioff = (_h + _wh) * w + (_w + _ww);
                    int woff = _wh * ww + _ww;
                    float lhs = in_plane[ioff] - in_zero_point;
                    float rhs = weight_plane[woff] -
                        (quantized ? weight_zero_point[0] : 0);
                    acc += lhs * rhs;
                  }
                }
              }
            } else {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const IT* in_plane = in_batch + _ic * h * w;
                const WT* weight_plane = weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    if (((_h + d0 * _wh - p0) >= 0) &&
                        ((_h + d0 * _wh - p0) < h) &&
                        ((_w + d1 * _ww - p1) >= 0) &&
                        ((_w + d1 * _ww - p1 < w))) {
                      int ioff =
                          (_h + d0 * _wh - p0) * w + (_w + d1 * _ww - p1);
                      int woff = _wh * ww + _ww;
                      float lhs = in_plane[ioff] - in_zero_point;
                      float rhs = weight_plane[woff] -
                          (quantized ? weight_zero_point[0] : 0);
                      acc += lhs * rhs;
                    }
                  }
                }
              }
            }
            if (quantized) {
              float val =
                  (per_tensor_quantized ? bias_scale[0] : bias_scale[_oc]) *
                  acc;
              out_plane[_oh * ow + _ow] =
                  kernels::quantize<OT>(val, inv_out_scale, out_zero_point);
            } else {
              out_plane[_oh * ow + _ow] = acc;
            }
          }
        }
      }
    }
  }
}

// The quantized convolution kernel. in_scale and weight_scale are implicit in
// bias_scale, since it is a product of the two. The kernel will branch to
// quantized::conv1d or quantized::conv2d based on the dimensionality of
// activation tensor.
void quantized_conv_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    exec_aten::IntArrayRef stride,
    exec_aten::IntArrayRef padding,
    exec_aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    bool channel_last,
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

  // Bool flag to check if weight tensor is quantized per-tensor or
  // per-channel
  bool per_tensor_quantized = bias_scale.numel() == 1;

  conv2d_nchw_core_generic<uint8_t, uint8_t, int32_t, uint8_t, true>(
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
      in_zero_point,
      weight_zero_point.const_data_ptr<int32_t>(),
      bias_scale.const_data_ptr<float>(),
      output_scale,
      (uint8_t)output_zero_point,
      per_tensor_quantized);
}

}; // namespace native
}; // namespace reference
}; // namespace impl

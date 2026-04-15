/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_conv2d.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// This implements a generic 2D float32 convolution kernel.
// The input is of shape [n x c x h x w] (batch x channels x height x width)
// The weight is of shape [oc x wc x wh x ww] (out_channels x weight_channels x
// weight_height x weight_width) The output is of shape [n x oc x oh x ow]
// (batch x out_channels x out_height x out_width) The bias is of shape [oc]

Tensor& conv2d_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    Tensor& out) {
  // Extract dimensions
  const int n = input.size(0);
  const int c = input.size(1);
  const int h = input.size(2);
  const int w = input.size(3);
  const int oc = weight.size(0);
  const int wc = weight.size(1);
  const int wh = weight.size(2);
  const int ww = weight.size(3);
  const int oh = out.size(2);
  const int ow = out.size(3);

  const int16_t s0 = static_cast<int16_t>(stride[0]);
  const int16_t s1 = static_cast<int16_t>(stride[1]);
  const int16_t p0 = static_cast<int16_t>(padding[0]);
  const int16_t p1 = static_cast<int16_t>(padding[1]);
  const int16_t d0 = static_cast<int16_t>(dilation[0]);
  const int16_t d1 = static_cast<int16_t>(dilation[1]);
  const int16_t g = static_cast<int16_t>(groups);

  const float* p_in = input.const_data_ptr<float>();
  const float* p_weight = weight.const_data_ptr<float>();
  const float* p_bias = bias.const_data_ptr<float>();
  float* p_out = out.mutable_data_ptr<float>();

  const bool zero_pad_unit_dilation = d0 == 1 && d1 == 1 && p0 == 0 && p1 == 0;
  const int ocpg = oc / g;
  const int icpg = c / g;

  for (int _n = 0; _n < n; ++_n) {
    const float* in_batch = p_in + _n * c * h * w;
    float* out_batch = p_out + _n * oc * oh * ow;
    for (int _g = 0; _g < g; ++_g) {
      int sic = _g * icpg;
      int soc = _g * ocpg;
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        float* out_plane = out_batch + _oc * oh * ow;
        const float* weight_batch = p_weight + _oc * wc * wh * ww;
        for (int _h = 0, _oh = 0; _oh < oh; _h += s0, ++_oh) {
          for (int _w = 0, _ow = 0; _ow < ow; _w += s1, ++_ow) {
            float acc = p_bias[_oc];
            if (zero_pad_unit_dilation) {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const float* in_plane = in_batch + _ic * h * w;
                const float* weight_plane =
                    weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int ioff = (_h + _wh) * w + (_w + _ww);
                    int woff = _wh * ww + _ww;
                    acc += in_plane[ioff] * weight_plane[woff];
                  }
                }
              }
            } else {
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                const float* in_plane = in_batch + _ic * h * w;
                const float* weight_plane =
                    weight_batch + (_ic - sic) * wh * ww;
                for (int _wh = 0; _wh < wh; ++_wh) {
                  for (int _ww = 0; _ww < ww; ++_ww) {
                    int h_pos = _h + d0 * _wh - p0;
                    int w_pos = _w + d1 * _ww - p1;
                    if (h_pos >= 0 && h_pos < h && w_pos >= 0 && w_pos < w) {
                      int ioff = h_pos * w + w_pos;
                      int woff = _wh * ww + _ww;
                      acc += in_plane[ioff] * weight_plane[woff];
                    }
                  }
                }
              }
            }
            out_plane[_oh * ow + _ow] = acc;
          }
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl

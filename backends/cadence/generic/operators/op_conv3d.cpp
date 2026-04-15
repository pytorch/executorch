/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_conv3d.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// This implements a generic 3D float32 convolution kernel.
// The input is of shape [n x c x d x h x w] (batch x channels x depth x height
// x width) The weight is of shape [oc x wc x wd x wh x ww] (out_channels x
// weight_channels x weight_depth x weight_height x weight_width) The output is
// of shape [n x oc x od x oh x ow] (batch x out_channels x out_depth x
// out_height x out_width) The bias is of shape [oc]

Tensor& conv3d_out(
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
  const int d = input.size(2);
  const int h = input.size(3);
  const int w = input.size(4);
  const int oc = weight.size(0);
  const int wc = weight.size(1);
  const int wd = weight.size(2);
  const int wh = weight.size(3);
  const int ww = weight.size(4);
  const int od = out.size(2);
  const int oh = out.size(3);
  const int ow = out.size(4);

  const int16_t s0 = static_cast<int16_t>(stride[0]);
  const int16_t s1 = static_cast<int16_t>(stride[1]);
  const int16_t s2 = static_cast<int16_t>(stride[2]);
  const int16_t p0 = static_cast<int16_t>(padding[0]);
  const int16_t p1 = static_cast<int16_t>(padding[1]);
  const int16_t p2 = static_cast<int16_t>(padding[2]);
  const int16_t d0 = static_cast<int16_t>(dilation[0]);
  const int16_t d1 = static_cast<int16_t>(dilation[1]);
  const int16_t d2 = static_cast<int16_t>(dilation[2]);
  const int16_t g = static_cast<int16_t>(groups);

  const float* p_in = input.const_data_ptr<float>();
  const float* p_weight = weight.const_data_ptr<float>();
  const float* p_bias = bias.const_data_ptr<float>();
  float* p_out = out.mutable_data_ptr<float>();

  const bool zero_pad_unit_dilation =
      d0 == 1 && d1 == 1 && d2 == 1 && p0 == 0 && p1 == 0 && p2 == 0;
  const int ocpg = oc / g;
  const int icpg = c / g;

  for (int _n = 0; _n < n; ++_n) {
    const float* in_batch = p_in + _n * c * d * h * w;
    float* out_batch = p_out + _n * oc * od * oh * ow;
    for (int _g = 0; _g < g; ++_g) {
      int sic = _g * icpg;
      int soc = _g * ocpg;
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        float* out_volume = out_batch + _oc * od * oh * ow;
        const float* weight_batch = p_weight + _oc * wc * wd * wh * ww;
        for (int _d = 0, _od = 0; _od < od; _d += s0, ++_od) {
          for (int _h = 0, _oh = 0; _oh < oh; _h += s1, ++_oh) {
            for (int _w = 0, _ow = 0; _ow < ow; _w += s2, ++_ow) {
              float acc = p_bias[_oc];
              if (zero_pad_unit_dilation) {
                for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                  const float* in_volume = in_batch + _ic * d * h * w;
                  const float* weight_volume =
                      weight_batch + (_ic - sic) * wd * wh * ww;
                  for (int _wd = 0; _wd < wd; ++_wd) {
                    for (int _wh = 0; _wh < wh; ++_wh) {
                      for (int _ww = 0; _ww < ww; ++_ww) {
                        int ioff =
                            (_d + _wd) * h * w + (_h + _wh) * w + (_w + _ww);
                        int woff = _wd * wh * ww + _wh * ww + _ww;
                        acc += in_volume[ioff] * weight_volume[woff];
                      }
                    }
                  }
                }
              } else {
                for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                  const float* in_volume = in_batch + _ic * d * h * w;
                  const float* weight_volume =
                      weight_batch + (_ic - sic) * wd * wh * ww;
                  for (int _wd = 0; _wd < wd; ++_wd) {
                    for (int _wh = 0; _wh < wh; ++_wh) {
                      for (int _ww = 0; _ww < ww; ++_ww) {
                        int d_pos = _d + d0 * _wd - p0;
                        int h_pos = _h + d1 * _wh - p1;
                        int w_pos = _w + d2 * _ww - p2;
                        if (d_pos >= 0 && d_pos < d && h_pos >= 0 &&
                            h_pos < h && w_pos >= 0 && w_pos < w) {
                          int ioff = d_pos * h * w + h_pos * w + w_pos;
                          int woff = _wd * wh * ww + _wh * ww + _ww;
                          acc += in_volume[ioff] * weight_volume[woff];
                        }
                      }
                    }
                  }
                }
              }
              out_volume[_od * oh * ow + _oh * ow + _ow] = acc;
            }
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

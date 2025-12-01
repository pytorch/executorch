/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_conv1d.h>

#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// This implements a generic 1D float32 convolution kernel.
// The input is of shape [n x c x w] (batch x channels x width)
// The weight is of shape [oc x wc x ww] (out_channels x weight_channels x
// weight_width) The output is of shape [n x oc x ow] (batch x out_channels x
// out_width) The bias is of shape [oc]

Tensor& conv1d_out(
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
  const int w = input.size(2);
  const int oc = weight.size(0);
  const int wc = weight.size(1);
  const int ww = weight.size(2);
  const int ow = out.size(2);

  const int16_t s = static_cast<int16_t>(stride[0]);
  const int16_t p = static_cast<int16_t>(padding[0]);
  const int16_t d = static_cast<int16_t>(dilation[0]);
  const int16_t g = static_cast<int16_t>(groups);

  const float* p_in = input.const_data_ptr<float>();
  const float* p_weight = weight.const_data_ptr<float>();
  const float* p_bias = bias.const_data_ptr<float>();
  float* p_out = out.mutable_data_ptr<float>();

  const bool zero_pad_unit_dilation = d == 1 && p == 0;
  const int ocpg = oc / g;
  const int icpg = c / g;

  for (int _n = 0; _n < n; ++_n) {
    const float* in_batch = p_in + _n * c * w;
    float* out_batch = p_out + _n * oc * ow;
    for (int _g = 0; _g < g; ++_g) {
      int sic = _g * icpg;
      int soc = _g * ocpg;
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        float* out_plane = out_batch + _oc * ow;
        const float* weight_batch = p_weight + _oc * wc * ww;
        for (int _w = 0, _ow = 0; _ow < ow; _w += s, ++_ow) {
          float acc = p_bias[_oc];
          if (zero_pad_unit_dilation) {
            for (int _ic = sic; _ic < sic + icpg; ++_ic) {
              const float* in_plane = in_batch + _ic * w;
              const float* weight_plane = weight_batch + (_ic - sic) * ww;
              for (int _ww = 0; _ww < ww; ++_ww) {
                int ioff = _w + _ww;
                acc += in_plane[ioff] * weight_plane[_ww];
              }
            }
          } else {
            for (int _ic = sic; _ic < sic + icpg; ++_ic) {
              const float* in_plane = in_batch + _ic * w;
              const float* weight_plane = weight_batch + (_ic - sic) * ww;
              for (int _ww = 0; _ww < ww; ++_ww) {
                int w_pos = _w + d * _ww - p;
                if (w_pos >= 0 && w_pos < w) {
                  acc += in_plane[w_pos] * weight_plane[_ww];
                }
              }
            }
          }
          out_plane[_ow] = acc;
        }
      }
    }
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl

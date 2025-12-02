/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "executorch/backends/cadence/generic/operators/op_transposed_convolution.h"

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

// This implements a generic 2d transposed_conv kernel that operates on raw
// pointers. The version handles both quantized and fp32 convolutions.
// The input is of shape [n x c x h x w]
// The weight is of shape [oc/groups x wc x wh x ww], where wc == c
// The output is of shape [n x oc x oh x ow]
// The bias is of shape [oc]
template <
    typename IT = float,
    typename WT = IT,
    typename BT = IT,
    typename OT = IT,
    bool quantized = false>
__attribute__((noinline)) void transposed_conv2d_nchw_core_generic(
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

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * c * h * w;
    OT* out_batch = p_out + _n * oc * oh * ow;
    // Compute separable transposed_convolution for each group
    for (int _g = 0; _g < groups; ++_g) {
      // Identify the input and output channels involved in the computation
      // of this group
      int sic = _g * icpg;
      int soc = _g * ocpg;

      // Populate all the output channels in the group
      for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
        OT* out_plane = out_batch + _oc * oh * ow;
        const WT* weight_batch = p_weight + (_oc - soc) * wc * wh * ww;
        // We compute one output channel at a time.
        for (int _oh = 0; _oh < oh; ++_oh) {
          for (int _ow = 0; _ow < ow; ++_ow) {
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to the
            // output channel being computed) with the corresponding weight
            // channel.
            for (int _ic = sic; _ic < sic + icpg; ++_ic) {
              const IT* in_plane = in_batch + _ic * h * w;
              const WT* weight_plane = weight_batch + _ic * wh * ww;
              for (int _wh = 0; _wh < wh; ++_wh) {
                int _ih = _oh - ((wh - 1) * d0) + _wh * d0 + p0;
                if (_ih < 0 || _ih >= s0 * h || _ih % s0 != 0) {
                  continue;
                }
                _ih = _ih / s0;
                for (int _ww = 0; _ww < ww; ++_ww) {
                  int _iw = _ow - ((ww - 1) * d1) + _ww * d1 + p1;
                  if (_iw < 0 || _iw >= s1 * w || _iw % s1 != 0) {
                    continue;
                  }
                  _iw = _iw / s1;
                  int ioff = _ih * w + _iw;
                  int woff = _wh * ww + _ww;
                  float lhs = in_plane[ioff] - in_zero_point;
                  float rhs = weight_plane[woff] -
                      (quantized ? weight_zero_point[0] : 0);
                  acc += lhs * rhs;
                }
              }
            }
            if (quantized) {
              float val =
                  (per_tensor_quantized ? bias_scale[0] : bias_scale[_oc]) *
                  acc;
              out_plane[_oh * ow + _ow] =
                  quantize<OT>(val, inv_out_scale, out_zero_point);
            } else {
              out_plane[_oh * ow + _ow] = acc;
            }
          }
        }
      }
    }
  }
}

template <
    typename IT = float,
    typename WT = IT,
    typename BT = IT,
    typename OT = IT,
    bool quantized = false>
__attribute__((noinline)) void transposed_conv2d_nhwc_core_generic(
    // All the arrays
    const IT* __restrict__ p_in,
    const WT* __restrict__ p_weight,
    const BT* __restrict__ p_bias,
    OT* __restrict__ p_out,
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

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * h * w * c;
    OT* out_batch = p_out + _n * oh * ow * oc;
    for (int _oh = 0; _oh < oh; ++_oh) {
      for (int _ow = 0; _ow < ow; ++_ow) {
        OT* out_line = out_batch + (_oh * ow + _ow) * oc;
        // Compute separable convolution for each group
        for (int _g = 0; _g < groups; ++_g) {
          // Identify the input and output channels involved in the computation
          // of this group
          int sic = _g * icpg;
          int soc = _g * ocpg;
          // Populate all the output channels in the group
          for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
            const WT* weight_batch = p_weight + (_oc - soc) * wh * ww * wc;
            // We compute one output channel at a time.
            float acc = p_bias[_oc];
            // Below is the stencil computation that performs the hadamard
            // product+accumulation of each input channel (contributing to
            // the output channel being computed) with the corresponding
            // weight channel.
            for (int _wh = 0; _wh < wh; ++_wh) {
              int _ih = _oh - ((wh - 1) * d0) + _wh * d0 + p0;
              if (_ih < 0 || _ih >= s0 * h || _ih % s0 != 0) {
                continue;
              }
              _ih = _ih / s0;
              for (int _ww = 0; _ww < ww; ++_ww) {
                int _iw = _ow - ((ww - 1) * d1) + _ww * d1 + p1;
                if (_iw < 0 || _iw >= s1 * w || _iw % s1 != 0) {
                  continue;
                }
                _iw = _iw / s1;
                const IT* in_line = in_batch + _ih * w * c + _iw * c;
                const WT* weight_line = weight_batch + _wh * ww * wc + _ww * wc;
                for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                  float lhs = in_line[_ic] - in_zero_point;
                  float rhs =
                      weight_line[_ic] - (quantized ? weight_zero_point[0] : 0);
                  acc += lhs * rhs;
                }
              }
            }
            if (quantized) {
              float val =
                  (per_tensor_quantized ? bias_scale[0] : bias_scale[_oc]) *
                  acc;
              out_line[_oc] = quantize<OT>(val, inv_out_scale, out_zero_point);
            } else {
              out_line[_oc] = acc;
            }
          }
        }
      }
    }
  }
}

void transposed_convolution_nchw(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    Tensor& output) {
  bool conv1d = input.dim() == 3;
  // input = [n, c, h, w]
  const int n = input.size(0);
  const int c = input.size(1);
  const int h = conv1d ? 1 : input.size(2);
  const int w = conv1d ? input.size(2) : input.size(3);
  // weight = [oc/groups, wc, wh, ww]
  const int wc = weight.size(1);
  const int wh = conv1d ? 1 : weight.size(2);
  const int ww = conv1d ? weight.size(2) : weight.size(3);
  // output = [n, oc, oh, ow]
  const int oc = output.size(1);
  const int oh = conv1d ? 1 : output.size(2);
  const int ow = conv1d ? output.size(2) : output.size(3);

  float* __restrict__ p_out = output.mutable_data_ptr<float>();
  const float* __restrict__ p_in = input.const_data_ptr<float>();
  const float* __restrict__ p_weight = weight.const_data_ptr<float>();
  const float* __restrict__ p_bias = bias.const_data_ptr<float>();

  transposed_conv2d_nchw_core_generic<>(
      p_in,
      p_weight,
      p_bias,
      p_out,
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
      conv1d ? 1 : stride[0],
      conv1d ? stride[0] : stride[1],
      conv1d ? 0 : padding[0],
      conv1d ? padding[0] : padding[1],
      conv1d ? 1 : dilation[0],
      conv1d ? dilation[0] : dilation[1],
      groups);
}

void transposed_convolution_nhwc(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    Tensor& output) {
  bool conv1d = input.dim() == 3;
  // input = [n, h, w, c]
  const int n = input.size(0);
  const int h = conv1d ? 1 : input.size(1);
  const int w = conv1d ? input.size(1) : input.size(2);
  const int c = conv1d ? input.size(2) : input.size(3);

  // weight = [oc/groups, wh, ww, wc]
  const int wh = conv1d ? 1 : weight.size(1);
  const int ww = conv1d ? weight.size(1) : weight.size(2);
  const int wc = conv1d ? weight.size(2) : weight.size(3);

  // output = [n, oh, ow, oc]
  const int oc = conv1d ? output.size(2) : output.size(3);
  const int oh = conv1d ? 1 : output.size(1);
  const int ow = conv1d ? output.size(1) : output.size(2);

  float* __restrict__ p_out = output.mutable_data_ptr<float>();
  const float* __restrict__ p_in = input.const_data_ptr<float>();
  const float* __restrict__ p_weight = weight.const_data_ptr<float>();
  const float* __restrict__ p_bias = bias.const_data_ptr<float>();

  transposed_conv2d_nhwc_core_generic<>(
      p_in,
      p_weight,
      p_bias,
      p_out,
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
      conv1d ? 1 : stride[0],
      conv1d ? stride[0] : stride[1],
      conv1d ? 0 : padding[0],
      conv1d ? padding[0] : padding[1],
      conv1d ? 1 : dilation[0],
      conv1d ? dilation[0] : dilation[1],
      groups);
}

Tensor& transposed_convolution_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    ET_UNUSED IntArrayRef output_padding,
    int64_t groups,
    bool channel_last,
    Tensor& output) {
  if (channel_last) {
    transposed_convolution_nhwc(
        input, weight, bias, stride, padding, dilation, groups, output);
  } else {
    transposed_convolution_nchw(
        input, weight, bias, stride, padding, dilation, groups, output);
  }

  return output;
}

// The quantized transposed_convolution kernel. in_scale and weight_scale are
// implicit in bias_scale, since it is a product of the two. The kernel will
// branch to quantized::conv1d or quantized::conv2d based on the dimensionality
// of activation tensor.
void quantized_transposed_conv_nchw(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    int32_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    float output_scale,
    int32_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& out) {
  bool conv1d = input.dim() == 3;
  // input = [n, c, h, w]
  const int n = input.size(0);
  const int c = input.size(1);
  const int h = conv1d ? 1 : input.size(2);
  const int w = conv1d ? input.size(2) : input.size(3);
  // weight = [oc/groups, wc, wh, ww]
  const int wc = weight.size(1);
  const int wh = conv1d ? 1 : weight.size(2);
  const int ww = conv1d ? weight.size(2) : weight.size(3);
  // output = [n, oc, oh, ow]
  const int oc = out.size(1);
  const int oh = conv1d ? 1 : out.size(2);
  const int ow = conv1d ? out.size(2) : out.size(3);

  ScalarType out_dtype = out.scalar_type();
  ScalarType weight_dtype = weight.scalar_type();
  // Bool flag to check if weight tensor is quantized per-tensor or
  // per-channel
  bool per_tensor_quantized = bias_scale.numel() == 1;

#define typed_quantized_conv2d_core(w_type, o_type)                            \
  transposed_conv2d_nchw_core_generic<uint8_t, w_type, int32_t, o_type, true>( \
      input.const_data_ptr<uint8_t>(),                                         \
      weight.const_data_ptr<w_type>(),                                         \
      bias.const_data_ptr<int32_t>(),                                          \
      out.mutable_data_ptr<o_type>(),                                          \
      n,                                                                       \
      c,                                                                       \
      h,                                                                       \
      w,                                                                       \
      oc,                                                                      \
      wc,                                                                      \
      wh,                                                                      \
      ww,                                                                      \
      oh,                                                                      \
      ow,                                                                      \
      stride[0],                                                               \
      stride[1],                                                               \
      padding[0],                                                              \
      padding[1],                                                              \
      dilation[0],                                                             \
      dilation[1],                                                             \
      groups,                                                                  \
      in_zero_point,                                                           \
      weight_zero_point.const_data_ptr<int32_t>(),                             \
      bias_scale.const_data_ptr<float>(),                                      \
      output_scale,                                                            \
      (o_type)output_zero_point,                                               \
      per_tensor_quantized);

#define typed_weight_dtype(out_dtype)                  \
  switch (weight_dtype) {                              \
    case ScalarType::Byte: {                           \
      typed_quantized_conv2d_core(uint8_t, out_dtype); \
      break;                                           \
    }                                                  \
    default:                                           \
      ET_DCHECK_MSG(                                   \
          false,                                       \
          "Unhandled weight dtype %s",                 \
          torch::executor::toString(weight_dtype));    \
  }

  switch (out_dtype) {
    case ScalarType::Byte: {
      typed_weight_dtype(uint8_t);
      break;
    }
    default:
      ET_DCHECK_MSG(
          false,
          "Unhandled out dtype %s",
          torch::executor::toString(out_dtype));
  }

#undef typed_weight_dtype
#undef typed_quantized_conv2d_core
}

void quantized_transposed_conv_nhwc(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int16_t groups,
    int32_t in_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    float output_scale,
    int32_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& out) {
  bool conv1d = input.dim() == 3;
  // input = [n, h, w, c]
  const int n = input.size(0);
  const int h = conv1d ? 1 : input.size(1);
  const int w = conv1d ? input.size(1) : input.size(2);
  const int c = conv1d ? input.size(2) : input.size(3);
  // weight = [oc/groups, wh, ww, wc]
  const int wh = conv1d ? 1 : weight.size(1);
  const int ww = conv1d ? weight.size(1) : weight.size(2);
  const int wc = conv1d ? weight.size(2) : weight.size(3);
  // output = [n, oh, ow, oc]
  const int oc = conv1d ? out.size(2) : out.size(3);
  const int oh = conv1d ? 1 : out.size(1);
  const int ow = conv1d ? out.size(1) : out.size(2);

  ScalarType out_dtype = out.scalar_type();
  ScalarType weight_dtype = weight.scalar_type();
  // Bool flag to check if weight tensor is quantized per-tensor or
  // per-channel
  bool per_tensor_quantized = bias_scale.numel() == 1;

#define typed_quantized_conv2d_core(w_type, o_type)                            \
  transposed_conv2d_nhwc_core_generic<uint8_t, w_type, int32_t, o_type, true>( \
      input.const_data_ptr<uint8_t>(),                                         \
      weight.const_data_ptr<w_type>(),                                         \
      bias.const_data_ptr<int32_t>(),                                          \
      out.mutable_data_ptr<o_type>(),                                          \
      n,                                                                       \
      h,                                                                       \
      w,                                                                       \
      c,                                                                       \
      oc,                                                                      \
      wh,                                                                      \
      ww,                                                                      \
      wc,                                                                      \
      oh,                                                                      \
      ow,                                                                      \
      stride[0],                                                               \
      stride[1],                                                               \
      padding[0],                                                              \
      padding[1],                                                              \
      dilation[0],                                                             \
      dilation[1],                                                             \
      groups,                                                                  \
      in_zero_point,                                                           \
      weight_zero_point.const_data_ptr<int32_t>(),                             \
      bias_scale.const_data_ptr<float>(),                                      \
      output_scale,                                                            \
      (o_type)output_zero_point,                                               \
      per_tensor_quantized);

#define typed_weight_dtype(out_dtype)                  \
  switch (weight_dtype) {                              \
    case ScalarType::Byte: {                           \
      typed_quantized_conv2d_core(uint8_t, out_dtype); \
      break;                                           \
    }                                                  \
    default:                                           \
      ET_DCHECK_MSG(                                   \
          false,                                       \
          "Unhandled weight dtype %s",                 \
          torch::executor::toString(weight_dtype));    \
  }

  switch (out_dtype) {
    case ScalarType::Byte: {
      typed_weight_dtype(uint8_t);
      break;
    }
    default:
      ET_DCHECK_MSG(
          false,
          "Unhandled out dtype %s",
          torch::executor::toString(out_dtype));
  }

#undef typed_weight_dtype
#undef typed_quantized_conv2d_core
}

Tensor& quantized_transposed_conv_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    IntArrayRef output_padding,
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
  if (channel_last) {
    quantized_transposed_conv_nhwc(
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
        out_multiplier,
        out_shift,
        out);
  } else {
    quantized_transposed_conv_nchw(
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
        out_multiplier,
        out_shift,
        out);
  }

  return out;
}

} // namespace native
} // namespace generic
} // namespace impl

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_nlc.h>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

namespace {
using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

// This implements a generic 1d conv kernel that operates on raw pointers.
// The quantized version handles both quantized convolutions for 1D inputs.
// The input is of shape [n x w x c] (NLC format)
// The weight is of shape [oc x ww x wc], where wc == c
// The output is of shape [n x ow x oc]
// The bias is of shape [oc]

template <
    typename IT = float,
    typename WT = IT,
    typename BT = IT,
    typename OT = IT,
    bool quantized = false>
__attribute__((noinline)) void conv1d_nlc_core_generic(
    // All the arrays
    const IT* __restrict__ p_in,
    const WT* __restrict__ p_weight,
    const BT* __restrict__ p_bias,
    OT* __restrict__ p_out,
    // The array sizes
    int32_t n,
    int32_t w,
    int32_t c,
    int32_t oc,
    int32_t ww,
    int32_t wc,
    int32_t ow,
    // Stride
    int16_t s,
    // Padding
    int16_t p,
    // Dilation
    int16_t d,
    // Group for depthwise conv
    int16_t groups,
    // Optional args that are only relevant for quantized convolution
    // input zero point
    IT in_zero_point = 0,
    // weight zero point
    int32_t weight_zero_point = 0,
    float bias_scale = 1,
    float out_scale = 1,
    OT out_zero_point = 0) {
  float inv_out_scale = 1. / out_scale;
  bool zero_pad_unit_dilation = d == 1 && p == 0;

  // Compute the number of in and out channels per group
  const int ocpg = oc / groups;
  const int icpg = c / groups;

  // Iterate over all the output batches (i.e., n)
  for (int _n = 0; _n < n; ++_n) {
    const IT* in_batch = p_in + _n * w * c;
    OT* out_batch = p_out + _n * ow * oc;
    for (int _w = 0, _ow = 0; _ow < ow; _w += s, ++_ow) {
      OT* out_line = out_batch + _ow * oc;
      // Compute separable convolution for each group
      for (int _g = 0; _g < groups; ++_g) {
        // Identify the input and output channels involved in the computation
        // of this group
        int sic = _g * icpg;
        int soc = _g * ocpg;
        // Populate all the output channels in the group
        for (int _oc = soc; _oc < soc + ocpg; ++_oc) {
          const WT* weight_batch = p_weight + _oc * ww * wc;
          // We compute one output channel at a time. The computation can be
          // thought of as a stencil computation: we iterate over an input of
          // size w x icpg, with a stencil of size ww x icpg, to
          // compute an output channel of size ow x 1.
          float acc = p_bias[_oc];
          // Below is the stencil computation that performs the hadamard
          // product+accumulation of each input channel (contributing to
          // the output channel being computed) with the corresponding
          // weight channel. If the padding is 0, and dilation is 1, then
          // we can remove the unnecessary checks, and simplify the code
          // so that it can be vectorized by Tensilica compiler.
          if (zero_pad_unit_dilation) {
            for (int _ww = 0; _ww < ww; ++_ww) {
              const IT* in_line = in_batch + (_w + _ww) * c;
              const WT* weight_line = weight_batch + _ww * wc;
              for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                float lhs = in_line[_ic] - in_zero_point;
                float rhs = weight_line[_ic - sic] -
                    (quantized ? weight_zero_point : 0);
                acc += lhs * rhs;
              }
            }
          } else {
            for (int _ww = 0; _ww < ww; ++_ww) {
              if (((_w + d * _ww - p) >= 0) && ((_w + d * _ww - p) < w)) {
                const IT* in_line = in_batch + (_w + d * _ww - p) * c;
                const WT* weight_line = weight_batch + _ww * wc;
                for (int _ic = sic; _ic < sic + icpg; ++_ic) {
                  float lhs = in_line[_ic] - in_zero_point;
                  float rhs = weight_line[_ic - sic] -
                      (quantized ? weight_zero_point : 0);
                  acc += lhs * rhs;
                }
              }
            }
          }
          if (quantized) {
            float val = bias_scale * acc;
            out_line[_oc] = quantize<OT>(val, inv_out_scale, out_zero_point);
          } else {
            out_line[_oc] = acc;
          }
        }
      }
    }
  }
}

void quantized_conv1d_nlc(
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
  // input = [n, w, c]
  const int n = input.size(0);
  const int w = input.size(1);
  const int c = input.size(2);
  // weight = [oc, ww, wc]
  const int oc = weight.size(0);
  const int ww = weight.size(1);
  const int wc = weight.size(2);
  // output = [n, ow, oc]
  const int ow = out.size(1);

#define typed_quantized_conv1d_nlc(ctype, dtype)                 \
  case ScalarType::dtype: {                                      \
    conv1d_nlc_core_generic<ctype, ctype, int32_t, ctype, true>( \
        input.const_data_ptr<ctype>(),                           \
        weight.const_data_ptr<ctype>(),                          \
        bias.const_data_ptr<int32_t>(),                          \
        out.mutable_data_ptr<ctype>(),                           \
        n,                                                       \
        w,                                                       \
        c,                                                       \
        oc,                                                      \
        ww,                                                      \
        wc,                                                      \
        ow,                                                      \
        stride[0],                                               \
        padding[0],                                              \
        dilation[0],                                             \
        groups,                                                  \
        in_zero_point,                                           \
        weight_zero_point,                                       \
        bias_scale,                                              \
        output_scale,                                            \
        (ctype)output_zero_point);                               \
    break;                                                       \
  }
  ScalarType dtype = out.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_conv1d_nlc);
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_conv1d_nlc
}

} // namespace

// Public exported kernel functions

::executorch::aten::Tensor& quantized_conv1d_nlc_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& out) {
  (void)ctx;
  (void)out_multiplier;
  (void)out_shift;
  quantized_conv1d_nlc(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      static_cast<int16_t>(groups),
      static_cast<int32_t>(input_zero_point),
      weight_zero_point.const_data_ptr<int32_t>()[0],
      bias_scale.const_data_ptr<float>()[0],
      static_cast<float>(output_scale),
      static_cast<int32_t>(output_zero_point),
      out);
  return out;
}

::executorch::aten::Tensor& quantized_conv1d_nlc_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  (void)ctx;
  quantized_conv1d_nlc(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      static_cast<int16_t>(groups),
      static_cast<int32_t>(input_zero_point),
      static_cast<int32_t>(weight_zero_point),
      static_cast<float>(bias_scale),
      static_cast<float>(output_scale),
      static_cast<int32_t>(output_zero_point),
      out);
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl

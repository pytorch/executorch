/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fused_quant/op_convolution_channels_last.h>
#include <executorch/backends/cadence/fused_quant/quant_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace cadence {
namespace fused_quant {
namespace native {

using executorch::aten::IntArrayRef;
using executorch::aten::optional;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

namespace {

// Convolution kernel for NHWC input and OHWI weight layouts.
//   inp:    [N, H_in, W_in, C_in]          (NHWC)
//   weight: [C_out, kH, kW, C_in/groups]   (OHWI)
//   bias:   [C_out]
//   out:    [N, H_out, W_out, C_out]        (NHWC)
void conv2d_nhwc_kernel(
    const float* inp,
    const float* weight,
    const float* bias,
    float* out,
    int64_t N,
    int64_t C_in,
    int64_t H_in,
    int64_t W_in,
    int64_t C_out,
    int64_t kH,
    int64_t kW,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    int64_t H_out,
    int64_t W_out) {
  int64_t C_in_per_group = C_in / groups;
  int64_t C_out_per_group = C_out / groups;

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t oh = 0; oh < H_out; ++oh) {
      for (int64_t ow = 0; ow < W_out; ++ow) {
        for (int64_t g = 0; g < groups; ++g) {
          for (int64_t oc = 0; oc < C_out_per_group; ++oc) {
            int64_t oc_global = g * C_out_per_group + oc;
            float sum = bias ? bias[oc_global] : 0.0f;
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                int64_t ih = oh * stride_h - pad_h + kh * dil_h;
                int64_t iw = ow * stride_w - pad_w + kw * dil_w;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                  for (int64_t ic = 0; ic < C_in_per_group; ++ic) {
                    int64_t ic_global = g * C_in_per_group + ic;
                    // NHWC: inp[n][ih][iw][ic_global]
                    float inp_val =
                        inp[((n * H_in + ih) * W_in + iw) * C_in + ic_global];
                    // OHWI: weight[oc_global][kh][kw][ic]
                    float w_val = weight
                        [((oc_global * kH + kh) * kW + kw) * C_in_per_group +
                         ic];
                    sum += inp_val * w_val;
                  }
                }
              }
            }
            // NHWC: out[n][oh][ow][oc_global]
            out[((n * H_out + oh) * W_out + ow) * C_out + oc_global] = sum;
          }
        }
      }
    }
  }
}

} // namespace

Tensor& convolution_channels_last_out(
    KernelRuntimeContext& ctx,
    const Tensor& inp,
    const Tensor& weight,
    const optional<Tensor>& bias,
    // inp qparams
    const optional<Tensor>& inp_scale,
    const optional<Tensor>& inp_zero_point,
    ScalarType inp_dtype,
    int64_t inp_quant_min,
    int64_t inp_quant_max,
    optional<int64_t> inp_axis,
    // weight qparams
    const optional<Tensor>& weight_scale,
    const optional<Tensor>& weight_zero_point,
    ScalarType weight_dtype,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    optional<int64_t> weight_axis,
    // bias qparams
    const optional<Tensor>& bias_scale,
    const optional<Tensor>& bias_zero_point,
    ScalarType bias_dtype,
    int64_t bias_quant_min,
    int64_t bias_quant_max,
    optional<int64_t> bias_axis,
    // out qparams
    const optional<Tensor>& out_scale,
    const optional<Tensor>& out_zero_point,
    ScalarType out_dtype,
    int64_t out_quant_min,
    int64_t out_quant_max,
    optional<int64_t> out_axis,
    // conv params
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    Tensor& out) {
  // NHWC layout: [N, H_in, W_in, C_in]
  int64_t N = inp.size(0);
  int64_t H_in = inp.size(1);
  int64_t W_in = inp.size(2);
  int64_t C_in = inp.size(3);

  // OHWI layout: [C_out, kH, kW, C_in/groups]
  int64_t C_out = weight.size(0);
  int64_t kH = weight.size(1);
  int64_t kW = weight.size(2);

  int64_t stride_h = stride[0];
  int64_t stride_w = stride[1];
  int64_t pad_h = padding[0];
  int64_t pad_w = padding[1];
  int64_t dil_h = dilation[0];
  int64_t dil_w = dilation[1];

  int64_t H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) / stride_h + 1;
  int64_t W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) / stride_w + 1;

  int64_t inp_numel = inp.numel();
  int64_t weight_numel = weight.numel();
  int64_t out_numel = N * H_out * W_out * C_out;

  bool inp_quantized = inp_scale.has_value();
  bool weight_quantized = weight_scale.has_value();
  bool bias_quantized = bias_scale.has_value();
  bool out_quantized = out_scale.has_value();

  // Dequantize input if needed.
  std::vector<float> inp_buf;
  const float* const inp_float = [&]() -> const float* {
    if (!inp_quantized) {
      return inp.const_data_ptr<float>();
    }
    inp_buf.resize(inp_numel);
    QParams qp = extract_qparams(
        inp_scale, inp_zero_point, inp_quant_min, inp_quant_max, inp_axis, inp);
    FUSED_QUANT_DTYPE_SWITCH(
        inp.scalar_type(),
        scalar_t,
        dequantize_buffer(
            inp.const_data_ptr<scalar_t>(), inp_buf.data(), inp_numel, qp);)
    return inp_buf.data();
  }();

  // Dequantize weight if needed.
  std::vector<float> weight_buf;
  const float* const weight_float = [&]() -> const float* {
    if (!weight_quantized) {
      return weight.const_data_ptr<float>();
    }
    weight_buf.resize(weight_numel);
    QParams qp = extract_qparams(
        weight_scale,
        weight_zero_point,
        weight_quant_min,
        weight_quant_max,
        weight_axis,
        weight);
    FUSED_QUANT_DTYPE_SWITCH(weight.scalar_type(),
                             scalar_t,
                             dequantize_buffer(
                                 weight.const_data_ptr<scalar_t>(),
                                 weight_buf.data(),
                                 weight_numel,
                                 qp);)
    return weight_buf.data();
  }();

  // Dequantize bias if needed.
  bool has_bias = bias.has_value();
  std::vector<float> bias_buf;
  const float* bias_float = nullptr;
  if (has_bias) {
    const Tensor& bias_val = bias.value();
    if (bias_quantized) {
      int64_t bias_numel = bias_val.numel();
      bias_buf.resize(bias_numel);
      QParams qp = extract_qparams(
          bias_scale,
          bias_zero_point,
          bias_quant_min,
          bias_quant_max,
          bias_axis,
          bias_val);
      FUSED_QUANT_DTYPE_SWITCH(bias_val.scalar_type(),
                               scalar_t,
                               dequantize_buffer(
                                   bias_val.const_data_ptr<scalar_t>(),
                                   bias_buf.data(),
                                   bias_numel,
                                   qp);)
      bias_float = bias_buf.data();
    } else {
      bias_float = bias_val.const_data_ptr<float>();
    }
  }

  // Run convolution in float.
  if (out_quantized) {
    std::vector<float> result_float(out_numel);
    conv2d_nhwc_kernel(
        inp_float,
        weight_float,
        bias_float,
        result_float.data(),
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        H_out,
        W_out);

    QParams qp = extract_qparams(
        out_scale, out_zero_point, out_quant_min, out_quant_max, out_axis, out);
    FUSED_QUANT_DTYPE_SWITCH(out.scalar_type(),
                             scalar_t,
                             quantize_buffer(
                                 result_float.data(),
                                 out.mutable_data_ptr<scalar_t>(),
                                 out_numel,
                                 qp);)
  } else {
    conv2d_nhwc_kernel(
        inp_float,
        weight_float,
        bias_float,
        out.mutable_data_ptr<float>(),
        N,
        C_in,
        H_in,
        W_in,
        C_out,
        kH,
        kW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        groups,
        H_out,
        W_out);
  }

  return out;
}

} // namespace native
} // namespace fused_quant
} // namespace cadence

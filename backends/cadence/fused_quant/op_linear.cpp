/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fused_quant/op_linear.h>
#include <executorch/backends/cadence/fused_quant/quant_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace cadence {
namespace fused_quant {
namespace native {

using executorch::aten::optional;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

namespace {

void linear_kernel(
    const float* inp,
    const float* weight,
    const float* bias,
    float* out,
    int64_t num_rows,
    int64_t in_features,
    int64_t out_features) {
  for (int64_t r = 0; r < num_rows; ++r) {
    for (int64_t o = 0; o < out_features; ++o) {
      float sum = bias ? bias[o] : 0.0f;
      for (int64_t i = 0; i < in_features; ++i) {
        sum += inp[r * in_features + i] * weight[o * in_features + i];
      }
      out[r * out_features + o] = sum;
    }
  }
}

} // namespace

Tensor& linear_out(
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
    Tensor& out) {
  int64_t in_features = inp.size(inp.dim() - 1);
  int64_t out_features = weight.size(0);
  int64_t num_rows = inp.numel() / in_features;
  int64_t inp_numel = inp.numel();
  int64_t weight_numel = weight.numel();
  int64_t out_numel = num_rows * out_features;

  bool inp_quantized = inp_scale.has_value();
  bool weight_quantized = weight_scale.has_value();
  bool bias_quantized = bias_scale.has_value();
  bool out_quantized = out_scale.has_value();

  // Dequantize inp
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

  // Dequantize weight
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

  // Dequantize bias if present and quantized
  std::vector<float> bias_buf;
  const float* const bias_float = [&]() -> const float* {
    if (!bias.has_value()) {
      return nullptr;
    }
    const Tensor& b = bias.value();
    if (!bias_quantized) {
      return b.const_data_ptr<float>();
    }
    int64_t bias_numel = b.numel();
    bias_buf.resize(bias_numel);
    QParams qp = extract_qparams(
        bias_scale,
        bias_zero_point,
        bias_quant_min,
        bias_quant_max,
        bias_axis,
        b);
    FUSED_QUANT_DTYPE_SWITCH(
        b.scalar_type(),
        scalar_t,
        dequantize_buffer(
            b.const_data_ptr<scalar_t>(), bias_buf.data(), bias_numel, qp);)
    return bias_buf.data();
  }();

  // Linear + optional quantize
  if (out_quantized) {
    std::vector<float> result_float(out_numel);
    linear_kernel(
        inp_float,
        weight_float,
        bias_float,
        result_float.data(),
        num_rows,
        in_features,
        out_features);
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
    linear_kernel(
        inp_float,
        weight_float,
        bias_float,
        out.mutable_data_ptr<float>(),
        num_rows,
        in_features,
        out_features);
  }

  return out;
}

} // namespace native
} // namespace fused_quant
} // namespace cadence

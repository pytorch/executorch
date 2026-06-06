/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fused_quant/op_mul.h>
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

void mul_kernel(
    const float* inp,
    const float* other,
    float* out,
    int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = inp[i] * other[i];
  }
}

} // namespace

Tensor& mul_out(
    KernelRuntimeContext& ctx,
    const Tensor& inp,
    const Tensor& other,
    const optional<Tensor>& inp_scale,
    const optional<Tensor>& inp_zero_point,
    ScalarType inp_dtype,
    int64_t inp_quant_min,
    int64_t inp_quant_max,
    optional<int64_t> inp_axis,
    const optional<Tensor>& other_scale,
    const optional<Tensor>& other_zero_point,
    ScalarType other_dtype,
    int64_t other_quant_min,
    int64_t other_quant_max,
    optional<int64_t> other_axis,
    const optional<Tensor>& out_scale,
    const optional<Tensor>& out_zero_point,
    ScalarType out_dtype,
    int64_t out_quant_min,
    int64_t out_quant_max,
    optional<int64_t> out_axis,
    Tensor& out) {
  (void)ctx;
  (void)inp_dtype;
  (void)other_dtype;
  (void)out_dtype;

  int64_t numel = inp.numel();

  bool inp_quantized = inp_scale.has_value();
  bool other_quantized = other_scale.has_value();
  bool out_quantized = out_scale.has_value();

  std::vector<float> inp_buf;
  const float* const inp_float = [&]() -> const float* {
    if (!inp_quantized) {
      return inp.const_data_ptr<float>();
    }
    inp_buf.resize(numel);
    QParams qp = extract_qparams(
        inp_scale, inp_zero_point, inp_quant_min, inp_quant_max, inp_axis, inp);
    FUSED_QUANT_DTYPE_SWITCH(
        inp.scalar_type(),
        scalar_t,
        dequantize_buffer(
            inp.const_data_ptr<scalar_t>(), inp_buf.data(), numel, qp);)
    return inp_buf.data();
  }();

  std::vector<float> other_buf;
  const float* const other_float = [&]() -> const float* {
    if (!other_quantized) {
      return other.const_data_ptr<float>();
    }
    other_buf.resize(numel);
    QParams qp = extract_qparams(
        other_scale,
        other_zero_point,
        other_quant_min,
        other_quant_max,
        other_axis,
        other);
    FUSED_QUANT_DTYPE_SWITCH(
        other.scalar_type(),
        scalar_t,
        dequantize_buffer(
            other.const_data_ptr<scalar_t>(), other_buf.data(), numel, qp);)
    return other_buf.data();
  }();

  if (out_quantized) {
    std::vector<float> result_float(numel);
    mul_kernel(inp_float, other_float, result_float.data(), numel);

    QParams qp = extract_qparams(
        out_scale, out_zero_point, out_quant_min, out_quant_max, out_axis, out);
    FUSED_QUANT_DTYPE_SWITCH(
        out.scalar_type(),
        scalar_t,
        quantize_buffer(
            result_float.data(), out.mutable_data_ptr<scalar_t>(), numel, qp);)
  } else {
    mul_kernel(inp_float, other_float, out.mutable_data_ptr<float>(), numel);
  }

  return out;
}

} // namespace native
} // namespace fused_quant
} // namespace cadence

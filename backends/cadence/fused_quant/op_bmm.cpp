/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/fused_quant/op_bmm.h>
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

void bmm_kernel(
    const float* inp,
    const float* other,
    float* out,
    int64_t batch,
    int64_t M,
    int64_t K,
    int64_t N) {
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += inp[b * M * K + m * K + k] * other[b * K * N + k * N + n];
        }
        out[b * M * N + m * N + n] = sum;
      }
    }
  }
}

} // namespace

Tensor& bmm_out(
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
  int64_t batch = inp.size(0);
  int64_t M = inp.size(1);
  int64_t K = inp.size(2);
  int64_t N = other.size(2);
  int64_t inp_numel = inp.numel();
  int64_t other_numel = other.numel();
  int64_t out_numel = batch * M * N;

  bool inp_quantized = inp_scale.has_value();
  bool other_quantized = other_scale.has_value();
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

  // Dequantize other
  std::vector<float> other_buf;
  const float* const other_float = [&]() -> const float* {
    if (!other_quantized) {
      return other.const_data_ptr<float>();
    }
    other_buf.resize(other_numel);
    QParams qp = extract_qparams(
        other_scale,
        other_zero_point,
        other_quant_min,
        other_quant_max,
        other_axis,
        other);
    FUSED_QUANT_DTYPE_SWITCH(other.scalar_type(),
                             scalar_t,
                             dequantize_buffer(
                                 other.const_data_ptr<scalar_t>(),
                                 other_buf.data(),
                                 other_numel,
                                 qp);)
    return other_buf.data();
  }();

  // BMM in float, then optionally quantize output
  if (out_quantized) {
    std::vector<float> result_float(out_numel);
    bmm_kernel(inp_float, other_float, result_float.data(), batch, M, K, N);

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
    bmm_kernel(
        inp_float, other_float, out.mutable_data_ptr<float>(), batch, M, K, N);
  }

  return out;
}

} // namespace native
} // namespace fused_quant
} // namespace cadence

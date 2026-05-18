/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>

#include <executorch/backends/cadence/fused_quant/op_hardswish.h>
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

void hardswish_kernel(const float* inp, float* out, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    float x = inp[i];
    out[i] = x * std::min(std::max(x + 3.0f, 0.0f), 6.0f) / 6.0f;
  }
}

} // namespace

Tensor& hardswish_out(
    KernelRuntimeContext& ctx,
    const Tensor& inp,
    const optional<Tensor>& inp_scale,
    const optional<Tensor>& inp_zero_point,
    ScalarType inp_dtype,
    int64_t inp_quant_min,
    int64_t inp_quant_max,
    optional<int64_t> inp_axis,
    const optional<Tensor>& out_scale,
    const optional<Tensor>& out_zero_point,
    ScalarType out_dtype,
    int64_t out_quant_min,
    int64_t out_quant_max,
    optional<int64_t> out_axis,
    Tensor& out) {
  int64_t numel = inp.numel();

  bool inp_quantized = inp_scale.has_value();
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

  if (out_quantized) {
    std::vector<float> result_float(numel);
    hardswish_kernel(inp_float, result_float.data(), numel);

    QParams qp = extract_qparams(
        out_scale, out_zero_point, out_quant_min, out_quant_max, out_axis, out);
    FUSED_QUANT_DTYPE_SWITCH(
        out.scalar_type(),
        scalar_t,
        quantize_buffer(
            result_float.data(), out.mutable_data_ptr<scalar_t>(), numel, qp);)
  } else {
    hardswish_kernel(inp_float, out.mutable_data_ptr<float>(), numel);
  }

  return out;
}

} // namespace native
} // namespace fused_quant
} // namespace cadence

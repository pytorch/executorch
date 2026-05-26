/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

// cppcheck-suppress unusedFunction
Tensor& quantized_activation_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& lut,
    Tensor& out) {
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Char,
      "quantized_activation: input must be int8");
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Char,
      "quantized_activation: output must be int8");
  ET_CHECK_MSG(
      lut.scalar_type() == ScalarType::Char,
      "quantized_activation: lut must be int8");
  ET_CHECK_MSG(
      lut.numel() == 256,
      "quantized_activation: lut must have 256 entries, got %" PRId64,
      static_cast<int64_t>(lut.numel()));
  ET_CHECK_MSG(
      input.numel() == out.numel(),
      "quantized_activation: input and output must have the same numel");

  const int8_t* in_data = input.const_data_ptr<int8_t>();
  const int8_t* lut_data = lut.const_data_ptr<int8_t>();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();

  // Bias the signed int8 input by 128 to use it as an unsigned table index;
  // the LUT entries are precomputed AoT from the input/output qparams and the
  // activation function (sigmoid / tanh / silu / ...), so the kernel does not
  // need to know which activation it is implementing.
  const int64_t n = input.numel();
  for (int64_t i = 0; i < n; ++i) {
    out_data[i] = lut_data[static_cast<uint8_t>(in_data[i] + 128)];
  }

  return out;
}

} // namespace native
} // namespace cortex_m

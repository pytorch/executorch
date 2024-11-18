/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/kernel_util/make_boxed_from_unboxed_functor.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/runtime.h>

#include <torch/library.h>

namespace torch {
namespace executor {

namespace native {

Tensor& quantize_per_token_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out);

Tensor& quantize_per_token_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  ::torch::executor::runtime_init();
  quantize_per_token_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
  return out;
}

at::Tensor quantize_per_token_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    c10::ScalarType dtype) {
  auto sizes = input.sizes().vec();
  auto output = at::zeros(sizes, dtype);
  TORCH_CHECK(dtype == c10::ScalarType::Char, "dtype must be char");
  WRAP_TO_ATEN(quantize_per_token_out_no_context, 6)
  (input, scale, zero_point, quant_min, quant_max, ScalarType::Char, output);
  return output;
}

Tensor& dequantize_per_token_out(
    RuntimeContext& context,
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out);

Tensor& dequantize_per_token_out_no_context(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    ScalarType out_dtype,
    Tensor& out) {
  exec_aten::RuntimeContext context{};
  ::torch::executor::runtime_init();
  dequantize_per_token_out(
      context,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      dtype,
      out_dtype,
      out);
  return out;
}

at::Tensor dequantize_per_token_aten(
    const at::Tensor& input,
    const at::Tensor& scale,
    const at::Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    c10::ScalarType dtype,
    c10::ScalarType out_dtype) {
  auto sizes = input.sizes().vec();
  auto output = at::zeros(sizes, out_dtype);
  TORCH_CHECK(dtype == c10::ScalarType::Char, "dtype must be char");
  TORCH_CHECK(out_dtype == c10::ScalarType::Float, "out_dtype must be float");
  WRAP_TO_ATEN(dequantize_per_token_out_no_context, 7)
  (input,
   scale,
   zero_point,
   quant_min,
   quant_max,
   ScalarType::Char,
   ScalarType::Float,
   output);
  return output;
}

} // namespace native
} // namespace executor
} // namespace torch

TORCH_LIBRARY(et_quant_test, m) {
  m.def(
      "quantize_per_token(Tensor input, Tensor scale, Tensor zero_points, int quant_min, int quant_max, ScalarType dtype) -> Tensor");
  m.def(
      "dequantize_per_token(Tensor input, Tensor scale, Tensor zero_points, int quant_min, int quant_max, ScalarType dtype, ScalarType out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(et_quant_test, CompositeExplicitAutograd, m) {
  m.impl(
      "quantize_per_token", torch::executor::native::quantize_per_token_aten);
  m.impl(
      "dequantize_per_token",
      torch::executor::native::dequantize_per_token_aten);
}

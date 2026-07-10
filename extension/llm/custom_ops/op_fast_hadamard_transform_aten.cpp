/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>
#include <executorch/extension/llm/custom_ops/op_fast_hadamard_transform.h>

#include <torch/library.h>

namespace torch::executor::native {
namespace {
template <typename EType, typename AType>
auto to_et_arg(AType&& value) {
  return executorch::extension::internal::type_convert<AType, EType>(
      std::forward<AType>(value));
}

at::Tensor& copy_et_result_to_out(Tensor& et_result, at::Tensor& out) {
  auto converted_result =
      executorch::extension::internal::type_convert<Tensor&, at::Tensor>(
          et_result)
          .call();
  at::native::resize_output(out, converted_result.sizes());
  out.copy_(converted_result);
  return out;
}

Tensor& fast_hadamard_transform_out_no_context(const Tensor& vec, Tensor& out) {
  executorch::aten::RuntimeContext context;
  return fast_hadamard_transform_out(context, vec, out);
}

at::Tensor& fast_hadamard_transform_out_aten(
    const at::Tensor& vec,
    at::Tensor& out) {
  auto vec_et = to_et_arg<Tensor>(vec);
  auto out_et = to_et_arg<Tensor&>(out);
  auto& et_result =
      fast_hadamard_transform_out_no_context(vec_et.call(), out_et.call());
  return copy_et_result_to_out(et_result, out);
}

at::Tensor fast_hadamard_transform_aten(const at::Tensor& vec) {
  auto out = at::empty_like(vec);
  fast_hadamard_transform_out_aten(vec, out);
  return out;
}
} // namespace
} // namespace torch::executor::native

TORCH_LIBRARY_FRAGMENT(llama, m) {
  m.def("fast_hadamard_transform(Tensor mat) -> Tensor");
  m.def(
      "fast_hadamard_transform.out(Tensor mat, *, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl(
      "fast_hadamard_transform",
      torch::executor::native::fast_hadamard_transform_aten);
  m.impl(
      "fast_hadamard_transform.out",
      torch::executor::native::fast_hadamard_transform_out_aten);
}

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
Tensor& fast_hadamard_transform_out_no_context(const Tensor& vec, Tensor& out) {
  exec_aten::RuntimeContext context;
  return fast_hadamard_transform_out(context, vec, out);
}
at::Tensor fast_hadamard_transform_aten(const at::Tensor& vec) {
  auto out = at::empty_like(vec);
  WRAP_TO_ATEN(fast_hadamard_transform_out_no_context, 1)
  (vec, out);
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
      WRAP_TO_ATEN(
          torch::executor::native::fast_hadamard_transform_out_no_context, 1));
}

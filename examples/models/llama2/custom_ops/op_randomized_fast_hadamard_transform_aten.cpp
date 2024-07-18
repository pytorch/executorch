/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/custom_ops/op_randomized_fast_hadamard_transform.h>
#include <executorch/extension/aten_util/make_aten_functor_from_et_functor.h>

#include <torch/library.h>

namespace torch::executor::native {
Tensor& randomized_fast_hadamard_transform_out_no_context(
    const Tensor& vec,
    const Tensor& s,
    Tensor& out) {
  exec_aten::RuntimeContext context;
  return randomized_fast_hadamard_transform_out(context, vec, s, out);
}
at::Tensor randomized_fast_hadamard_transform_aten(
    const at::Tensor& vec,
    const at::Tensor& s) {
  auto out = at::empty_like(vec);
  WRAP_TO_ATEN(randomized_fast_hadamard_transform_out_no_context, 2)
  (vec, s, out);
  return out;
}
} // namespace torch::executor::native

TORCH_LIBRARY_FRAGMENT(llama, m) {
  m.def("randomized_fast_hadamard_transform(Tensor vec, Tensor s) -> Tensor");
  m.def(
      "randomized_fast_hadamard_transform.out(Tensor vec, Tensor s, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(llama, CompositeExplicitAutograd, m) {
  m.impl(
      "randomized_fast_hadamard_transform",
      torch::executor::native::randomized_fast_hadamard_transform_aten);
  m.impl(
      "randomized_fast_hadamard_transform.out",
      WRAP_TO_ATEN(
          torch::executor::native::
              randomized_fast_hadamard_transform_out_no_context,
          2));
}

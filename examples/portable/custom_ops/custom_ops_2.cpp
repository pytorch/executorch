/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>

namespace custom {
namespace native {

using at::Tensor;
using c10::ScalarType;

// mul4(Tensor input) -> Tensor
Tensor mul4_impl(const Tensor& in) {
  // naive approach
  at::Tensor out = at::zeros_like(in);
  out.copy_(in);
  out.mul_(4);
  return out;
}

TORCH_LIBRARY_FRAGMENT(my_ops, m) {
  m.def("my_ops::mul4(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CompositeExplicitAutograd, m) {
  m.impl("mul4", TORCH_FN(mul4_impl));
}
} // namespace native
} // namespace custom

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h> // @manual

std::tuple<at::Tensor, at::Tensor> awesome_op_impl(const at::Tensor& input) {
  return std::make_tuple(input, at::Tensor());
}

std::tuple<at::Tensor, at::Tensor>
awesome_op_out(const at::Tensor& input, at::Tensor& out1, at::Tensor& out2) {
  (void)out1;
  (void)out2;
  return std::make_tuple(input, at::Tensor());
}

TORCH_LIBRARY_FRAGMENT(my_awesome_3rdparty_ns, m) {
  m.def("my_awesome_op.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("my_awesome_op.func(Tensor input) -> Tensor");
  // schema mismatch test, missing default value in out variant
  m.def(
      "schema_mismatch_op.out(Tensor input, Scalar scalar, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("schema_mismatch_op(Tensor input, Scalar scalar=1) -> Tensor");
  m.def("awesome_op(Tensor input) -> (Tensor, Tensor)", awesome_op_impl);
  m.def(
      "awesome_op.out(Tensor input, Tensor(a!) out1, Tensor(b!) out2) -> (Tensor(a!), Tensor(b!))",
      awesome_op_out);
}

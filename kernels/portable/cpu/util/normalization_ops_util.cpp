/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

void check_batch_norm_args(
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out) {
  // All tensors must be the same dtype
  ET_CHECK_SAME_DTYPE3(in, running_mean, running_var);
  ET_CHECK_SAME_DTYPE2(in, out);
  if (weight.has_value()) {
    ET_CHECK_SAME_DTYPE2(in, weight.value());
  }
  if (bias.has_value()) {
    ET_CHECK_SAME_DTYPE2(in, bias.value());
  }

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  // All parameter tensors must be of dim 1 and have length equal to the
  // channels dim of in
  ET_CHECK(running_mean.dim() == 1 && running_mean.size(0) == in.size(C_dim));
  ET_CHECK(running_var.dim() == 1 && running_var.size(0) == in.size(C_dim));
  if (weight.has_value()) {
    ET_CHECK(
        weight.value().dim() == 1 && weight.value().size(0) == in.size(C_dim));
  }
  if (bias.has_value()) {
    ET_CHECK(bias.value().dim() == 1 && bias.value().size(0) == in.size(C_dim));
  }
}

} // namespace executor
} // namespace torch

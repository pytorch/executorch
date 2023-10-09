/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/activation_ops_util.h>

namespace torch {
namespace executor {

bool check_gelu_args(const Tensor& in, string_view approximate, Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      approximate == "tanh" || approximate == "none",
      "Invalid approximation format: %.*s for gelu",
      static_cast<int>(approximate.length()),
      approximate.data());
  return true;
}

} // namespace executor
} // namespace torch

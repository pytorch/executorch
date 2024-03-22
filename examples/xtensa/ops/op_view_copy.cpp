/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor&
view_copy_out(const Tensor& input, const at::IntArrayRef size, Tensor& out) {
  kernels::memcpy(
      out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

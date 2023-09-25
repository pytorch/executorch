/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/transpose_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using SizesType = exec_aten::SizesType;
using StridesType = exec_aten::StridesType;
using Tensor = exec_aten::Tensor;

namespace {

/**
 * Verifies preconditions of t_copy_int_out
 */
void check_preconditions(const Tensor& a, Tensor& out) {
  auto a_dim = a.dim();
  ET_CHECK_MSG(
      a_dim >= 0 && a_dim <= 2,
      "Rank of tensor a has to be <=2 but received tensor of rank : %zd.:",
      a_dim);
  if (a_dim < 2) {
    ET_CHECK_SAME_SHAPE_AND_DTYPE2(a, out);
  } else {
    ET_CHECK_SAME_DTYPE2(a, out);
    ET_CHECK_MSG(
        (a.sizes()[0] == out.sizes()[1]) && (a.sizes()[1] == out.sizes()[0]),
        "Input tensor and output tensor shapes do not support transposing");
    ET_CHECK_MSG(out.dim() == 2, "Output tensor must have same dim (2)");
  }
}

} // namespace

/**
 * Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
 * 0-D and 1-D tensors are returned as is. When input is a 2-D tensor this
 * is equivalent to transpose(input, 0, 1).
 * t_copy.out(Tensor self, Tensor(a!) out)
 */
Tensor& t_copy_out(RuntimeContext& ctx, const Tensor& a, Tensor& out) {
  (void)ctx;
  check_preconditions(a, out);
  int dim_1 = a.sizes().size() == 2 ? 1 : 0;
#define TRANSPOSE_TENSORS(ctype, dtype)         \
  case ScalarType::dtype:                       \
    transpose_tensors<ctype>(a, 0, dim_1, out); \
    break;

  switch (a.scalar_type()) {
    ET_FORALL_SCALAR_TYPES(TRANSPOSE_TENSORS)
    default:
      ET_CHECK_MSG(
          false, "Unhandled dtype %hhd", static_cast<int8_t>(a.scalar_type()));
  }

#undef TRANSPOSE_TENSORS

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

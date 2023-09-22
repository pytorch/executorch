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
 * Verifies preconditions of transpose_copy_int_out
 */
void check_preconditions(
    const Tensor& a,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  auto a_dim = a.dim();
  ET_CHECK_MSG(
      a_dim >= 0 && a_dim == out.dim(), "invalid rank of tensor a: %zd", a_dim);
  if (a_dim == 0) {
    ET_CHECK(dim0 == 0 || dim0 == -1);
    ET_CHECK(dim1 == 0 || dim1 == -1);
    return;
  }
  ET_CHECK_MSG(
      dim0 >= 0 && dim0 < a_dim,
      "dim0: %" PRId64 " out of bounds [0,%zd)",
      dim0,
      a_dim);
  ET_CHECK_MSG(
      dim1 >= 0 && dim1 < a_dim,
      "dim1: %" PRId64 " out of bounds [0,%zd)",
      dim1,
      a_dim);
  ET_CHECK_MSG(
      a_dim <= kTensorDimensionLimit,
      "input tensor rank %zd greater than %zu",
      a_dim,
      kTensorDimensionLimit);
}

} // namespace

/**
 * Swaps dimension 'dim0' of 'a' with 'dim1', and copying
 * that mutation into `out` in a manner such that the data is densely packed
 * and is_contiguous() would return true (stride dim[size-1] = 1).
 *
 * transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out)
 */
Tensor& transpose_copy_int_out(
    RuntimeContext& ctx,
    const Tensor& a,
    int64_t dim0,
    int64_t dim1,
    Tensor& out) {
  (void)ctx;

  ET_CHECK_SAME_DTYPE2(a, out);

  // fix python negative indexing
  if (dim0 < 0) {
    dim0 += out.dim();
  }
  if (dim1 < 0) {
    dim1 += out.dim();
  }
  check_preconditions(a, dim0, dim1, out);
#define TRANSPOSE_TENSORS(ctype, dtype)           \
  case ScalarType::dtype:                         \
    transpose_tensors<ctype>(a, dim0, dim1, out); \
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

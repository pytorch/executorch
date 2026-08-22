/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include <cstddef>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename CTYPE>
void diagonal_copy_impl(
    const Tensor& in,
    int64_t offset,
    int64_t dim1,
    int64_t dim2,
    Tensor& out) {
  if (out.numel() == 0) {
    return;
  }

  int64_t storage_offset = 0;
  size_t diag_size = out.size(out.dim() - 1);

  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += offset * in.strides().at(dim2);
  } else {
    storage_offset -= offset * in.strides().at(dim1);
  }

  size_t new_ndim = out.dim();
  int64_t new_sizes[kTensorDimensionLimit];
  for (const auto i : c10::irange(new_ndim)) {
    new_sizes[i] = out.size(i);
  }

  int64_t new_strides[kTensorDimensionLimit];
  size_t shift = 0;
  size_t in_dim = in.dim();
  for (const auto d : c10::irange(in_dim)) {
    if (static_cast<int64_t>(d) == dim1 || static_cast<int64_t>(d) == dim2) {
      shift++;
    } else {
      new_strides[d - shift] = in.strides().at(d);
    }
  }
  new_strides[in_dim - 2] = in.strides().at(dim1) + in.strides().at(dim2);

  as_strided_copy<CTYPE>(
      in, {new_sizes, new_ndim}, {new_strides, new_ndim}, storage_offset, out);
}

} // namespace

Tensor& diagonal_copy_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t offset,
    int64_t dim1,
    int64_t dim2,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_diagonal_copy_args(in, dim1, dim2, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  if (dim1 < 0) {
    dim1 += nonzero_dim(in);
  }
  if (dim2 < 0) {
    dim2 += nonzero_dim(in);
  }

  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_diagonal_copy_out_target_size(
      in, offset, dim1, dim2, expected_out_size, &expected_out_dim);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "diagonal_copy.out";

  ET_SWITCH_ALL_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
    diagonal_copy_impl<CTYPE>(in, offset, dim1, dim2, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

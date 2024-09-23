/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstddef>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_roll_args(
    const Tensor& in,
    IntArrayRef shifts,
    IntArrayRef dims,
    const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensor_has_rank_greater_or_equal_to(in, 1));
  if (in.numel() > 0) {
    for (const auto& d : dims) {
      ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(d, in.dim()));
    }
  }
  ET_LOG_AND_RETURN_IF_FALSE(!shifts.empty());
  ET_LOG_AND_RETURN_IF_FALSE(shifts.size() == dims.size());
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  return true;
}

size_t unshift_flat_ix(size_t ix, const Tensor& in, IntArrayRef dim_shifts) {
  size_t ix_coord[kTensorDimensionLimit];
  indexToCoordinate(in, ix, ix_coord);

  size_t shifted_coord[kTensorDimensionLimit];
  for (size_t d = 0; d < in.dim(); d++) {
    shifted_coord[d] =
        (ix_coord[d] + in.size(d) - dim_shifts[d] % in.size(d)) % in.size(d);
  }

  return coordinateToIndex(in, shifted_coord);
}

} // namespace

Tensor& roll_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef shifts,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, check_roll_args(in, shifts, dims, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  if (in.numel() == 0) {
    return out;
  }

  int64_t dim_shift_array[kTensorDimensionLimit];
  for (size_t i = 0; i < in.dim(); i++) {
    dim_shift_array[i] = 0;
  }
  for (size_t i = 0; i < dims.size(); i++) {
    const auto d = dims[i] < 0 ? dims[i] + in.dim() : dims[i];
    dim_shift_array[d] += shifts[i];
  }

  size_t dim_shift_array_length = static_cast<size_t>(in.dim()); // NOLINT
  IntArrayRef dim_shifts(dim_shift_array, dim_shift_array_length);

  constexpr auto name = "roll.out";

  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    for (size_t ix = 0; ix < out.numel(); ++ix) {
      out_data[ix] = in_data[unshift_flat_ix(ix, in, dim_shifts)];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

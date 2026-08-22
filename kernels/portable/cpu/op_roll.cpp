/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
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
  for (const auto i : c10::irange(in.dim())) {
    dim_shift_array[i] = 0;
  }
  for (const auto i : c10::irange(dims.size())) {
    const auto d = dims[i] < 0 ? dims[i] + in.dim() : dims[i];
    dim_shift_array[d] += shifts[i];
  }

  size_t dim_shift_array_length = static_cast<size_t>(in.dim()); // NOLINT
  IntArrayRef dim_shifts(dim_shift_array, dim_shift_array_length);

  static constexpr auto name = "roll.out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    const bool out_is_default = executorch::runtime::is_contiguous_dim_order(
        out.dim_order().data(), out.dim_order().size());

    for (const auto ix : c10::irange(out.numel())) {
      // @lint-ignore CLANGTIDY facebook-hte-CArray
      size_t coord[kTensorDimensionLimit];
      indexToCoordinate(in, ix, coord);

      // @lint-ignore CLANGTIDY facebook-hte-CArray
      size_t shifted_coord[kTensorDimensionLimit];
      for (const auto d : c10::irange(in.dim())) {
        shifted_coord[d] =
            (coord[d] + in.size(d) - dim_shifts[d] % in.size(d)) % in.size(d);
      }

      out_data[out_is_default ? ix : coordinateToIndex(out, coord)] =
          in_data[coordinateToIndex(in, shifted_coord)];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

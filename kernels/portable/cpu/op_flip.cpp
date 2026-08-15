/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {
namespace {

bool check_flip_args(const Tensor& in, IntArrayRef dims, const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  return check_dim_list_is_valid(in, dims);
}

} // namespace

Tensor& flip_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef dims,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, check_flip_args(in, dims, out), InvalidArgument, out);

  bool flip_dim_data[kTensorDimensionLimit];
  for (const auto i : c10::irange(in.dim())) {
    flip_dim_data[i] = false;
  }
  for (const auto i : c10::irange(dims.size())) {
    const auto d = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    flip_dim_data[d] = true;
  }
  size_t flip_dim_length = static_cast<size_t>(in.dim()); // NOLINT
  ArrayRef<bool> flip_dim(flip_dim_data, flip_dim_length);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "flip_out";

  ET_SWITCH_REALHBBF16_TYPES(in.scalar_type(), ctx, op_name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    const bool out_is_default = executorch::runtime::is_contiguous_dim_order(
        out.dim_order().data(), out.dim_order().size());

    for (const auto ix : c10::irange(in.numel())) {
      // @lint-ignore CLANGTIDY facebook-hte-CArray
      size_t coord[kTensorDimensionLimit];
      indexToCoordinate(in, ix, coord);

      // @lint-ignore CLANGTIDY facebook-hte-CArray
      size_t src_coord[kTensorDimensionLimit];
      for (const auto d : c10::irange(in.dim())) {
        src_coord[d] = flip_dim[d] ? in.size(d) - coord[d] - 1 : coord[d];
      }

      out_data[out_is_default ? ix : coordinateToIndex(out, coord)] =
          in_data[coordinateToIndex(in, src_coord)];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

size_t unflip_flat_ix(size_t ix, const Tensor& in, ArrayRef<bool> flip_dim) {
  size_t ix_coord[kTensorDimensionLimit];
  indexToCoordinate(in, ix, ix_coord);

  size_t unflip_coord[kTensorDimensionLimit];
  for (size_t d = 0; d < in.dim(); d++) {
    if (flip_dim[d]) {
      unflip_coord[d] = in.size(d) - ix_coord[d] - 1;
    } else {
      unflip_coord[d] = ix_coord[d];
    }
  }

  return coordinateToIndex(in, unflip_coord);
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
  for (size_t i = 0; i < in.dim(); i++) {
    flip_dim_data[i] = false;
  }
  for (size_t i = 0; i < dims.size(); i++) {
    const auto d = dims[i] < 0 ? dims[i] + nonzero_dim(in) : dims[i];
    flip_dim_data[d] = true;
  }
  size_t flip_dim_length = static_cast<size_t>(in.dim()); // NOLINT
  ArrayRef<bool> flip_dim(flip_dim_data, flip_dim_length);

  constexpr auto name = "flip.out";

  ET_SWITCH_REALHB_TYPES(in.scalar_type(), ctx, name, CTYPE, [&] {
    const CTYPE* in_data = in.const_data_ptr<CTYPE>();
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();

    for (size_t ix = 0; ix < out.numel(); ++ix) {
      out_data[ix] = in_data[unflip_flat_ix(ix, in, flip_dim)];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

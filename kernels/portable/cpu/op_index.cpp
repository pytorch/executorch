/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/advanced_index_util.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorOptList = exec_aten::ArrayRef<exec_aten::optional<Tensor>>;

Tensor& index_Tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    TensorOptList indices,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_index_args(in, indices, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  size_t block_count = count_index_blocks(indices);

  // If indices list is empty or all indices are null, just copy the input to
  // output and return early.
  if (block_count == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
    ET_SWITCH_REALHB_TYPES(in_type, ctx, "index.Tensor_out", CTYPE, [&]() {
      const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
      CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
      memcpy(out_data, in_data, in.nbytes());
    });
    return out;
  }

  // The output shape depends on whether all the non-null indices are adjacent
  // or not.
  bool adjacent = (block_count == 1);

  Tensor::SizesType expected_size[kTensorDimensionLimit];
  size_t expected_ndim = 0;

  ET_KERNEL_CHECK(
      ctx,
      get_index_out_target_size(
          in, indices, adjacent, expected_size, &expected_ndim),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_size, expected_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.numel() == 0) {
    return out;
  }

  int32_t dim_map[kTensorDimensionLimit];
  int32_t ix_map[kTensorDimensionLimit];
  size_t start = 0;
  size_t xdim = 0;

  if (adjacent) {
    start = get_num_leading_null_indices(indices);
  }
  xdim = get_indices_broadcast_ndim(indices);
  compute_dim_map(in, indices, dim_map, block_count == 1);
  compute_index_map(in, indices, ix_map);

  ET_SWITCH_REALHB_TYPES(in_type, ctx, "index.Tensor_out", CTYPE, [&]() {
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (auto out_ix = 0; out_ix < out.numel(); out_ix++) {
      size_t in_ix = 0;
      bool success = true;
      std::tie(in_ix, success) =
          get_in_ix(in, indices, out, out_ix, start, xdim, dim_map, ix_map);
      ET_KERNEL_CHECK(ctx, success, InvalidArgument, );
      out_data[out_ix] = in_data[in_ix];
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

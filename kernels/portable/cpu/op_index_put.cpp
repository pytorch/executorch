/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/advanced_index_util.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& index_put_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    exec_aten::ArrayRef<exec_aten::optional<Tensor>> indices,
    const Tensor& values,
    const bool accumulate,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_index_args(in, indices, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dtype(in, values), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  size_t block_count = count_index_blocks(indices);

  // If indices list is empty or all indices are null, then the operation is
  // performed over then entire input tensor. So, this is equivalent to
  // out = values when accumulate is false. Otherwise, the operation is
  // out = in + values where accumulate is true.
  if (block_count == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

    // Check that values tensors can be broadcasted to out
    ET_KERNEL_CHECK(
        ctx, tensor_is_broadcastable_to(values, out), InvalidArgument, out);

    ET_SWITCH_REALHB_TYPES(in_type, ctx, "index_put.out", CTYPE, [&]() {
      apply_binary_elementwise_fn<CTYPE, CTYPE, CTYPE>(
          [accumulate](const CTYPE val_in, const CTYPE val) {
            return accumulate ? val_in + val : val;
          },
          in,
          values,
          out);
    });
    return out;
  }

  // The index output shape depends on whether all the non-null indices are
  // adjacent or not.
  bool adjacent = (block_count == 1);

  // Compute the expected index output shape.
  Tensor::SizesType x_sizes[kTensorDimensionLimit];
  size_t x_dim = 0;
  ET_KERNEL_CHECK(
      ctx,
      get_index_out_target_size(in, indices, adjacent, x_sizes, &x_dim),
      InvalidArgument,
      out);

  // Check that values tensors can be broadcasted to indexing result
  ET_KERNEL_CHECK(
      ctx,
      tensor_is_broadcastable_to(values.sizes(), {x_sizes, x_dim}),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  // No further action if the input is empty
  if (in.numel() == 0) {
    return out;
  }

  // To start, copy the input data into the out tensor
  memcpy(out.mutable_data_ptr<char>(), in.const_data_ptr<char>(), in.nbytes());

  // In what follows, `x = in[indices]`. This tensor is implicit, and it would
  // be much easier to be able to allocate memory, and then call index.Tensor
  // to compute `x`. But since we can't do that, we have to keep track of its
  // shape, number of dimensions, number of elements, and use it to translate
  // coordinates from `x` to `in`.

  // Compute the dim_map and ix_map needed for `x -> in` coordinate translation
  int32_t dim_map[kTensorDimensionLimit];
  int32_t ix_map[kTensorDimensionLimit];
  size_t start = 0;

  if (adjacent) {
    start = get_num_leading_null_indices(indices);
  }
  size_t bc_ndim = get_indices_broadcast_ndim(indices);
  compute_dim_map(in, indices, dim_map, block_count == 1);
  compute_index_map(in, indices, ix_map);

  // Compute the number of elements in the indexed space
  size_t x_numel = 1;
  for (size_t i = 0; i < x_dim; i++) {
    x_numel *= x_sizes[i];
  }

  ET_SWITCH_REALHB_TYPES(in_type, ctx, "index_put.out", CTYPE, [&]() {
    const CTYPE* const values_data = values.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (auto x_ix = 0; x_ix < x_numel; x_ix++) {
      size_t in_ix = 0;

      size_t x_coord[kTensorDimensionLimit];
      delinearize_index(x_ix, {x_sizes, x_dim}, x_coord, kTensorDimensionLimit);

      size_t in_coord[kTensorDimensionLimit];

      ET_KERNEL_CHECK(
          ctx,
          get_in_coord(
              in, indices, start, bc_ndim, dim_map, ix_map, x_coord, in_coord),
          InvalidArgument, );

      in_ix = coordinateToIndex(in, in_coord);

      // Braodcast values
      size_t val_ix = linearize_access_indexes(x_coord, x_dim, values);
      if (accumulate) {
        out_data[in_ix] += values_data[val_ix];
      } else {
        out_data[in_ix] = values_data[val_ix];
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& masked_select_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mask,
    Tensor& out) {
  ScalarType in_type = in.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_realhbbf16_type(in),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, mask.scalar_type() == ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, out.scalar_type() == in_type, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, mask, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_are_broadcastable_between(in, mask), InvalidArgument, out);

  // If input or mask is empty, the output should be empty
  if (in.numel() == 0 || mask.numel() == 0) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, {0}) == Error::Ok, InvalidArgument, out);
    return out;
  }

  // Compute the shape resulting from broadcasting the mask against the input
  size_t broadcast_ndim = 0;
  Tensor::SizesType broadcast_sizes[kTensorDimensionLimit];
  Error err = get_broadcast_target_size(
      in, mask, broadcast_sizes, kTensorDimensionLimit, &broadcast_ndim);
  if (err != Error::Ok) {
    ET_KERNEL_CHECK_MSG(
        ctx, false, InvalidArgument, out, "Failed to broadcast input and mask");
  }
  size_t broadcast_numel = 1;
  for (size_t i = 0; i < broadcast_ndim; i++) {
    broadcast_numel *= broadcast_sizes[i];
  }

  // Compute the number of out elements
  size_t mask_true_count = 0;
  const bool* const mask_data = mask.const_data_ptr<bool>();
  for (size_t i = 0; i < mask.numel(); ++i) {
    if (mask_data[i]) {
      mask_true_count++;
    }
  }
  Tensor::SizesType out_numel =
      mask_true_count * (broadcast_numel / mask.numel());

  // Resize the out tensor
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {out_numel}) == Error::Ok, InvalidArgument, out);

  const char* const in_data =
      reinterpret_cast<const char*>(in.const_data_ptr());
  char* const out_data = reinterpret_cast<char*>(out.mutable_data_ptr());
  const auto elem_size = in.element_size();

  // Figure out if `in` is broadcasted
  bool in_is_broadcasted = false;
  if (in.dim() != broadcast_ndim) {
    in_is_broadcasted = true;
  } else {
    for (size_t i = 0; i < in.dim(); ++i) {
      if (in.size(i) != broadcast_sizes[i]) {
        in_is_broadcasted = true;
      }
    }
  }

  // Figure out if `mask` is broadcasted
  bool mask_is_broadcasted = false;
  if (mask.dim() != broadcast_ndim) {
    mask_is_broadcasted = true;
  } else {
    for (size_t i = 0; i < mask.dim(); ++i) {
      if (mask.size(i) != broadcast_sizes[i]) {
        mask_is_broadcasted = true;
      }
    }
  }

  // Figure out if either `in` or `mask` is broadcasted
  bool any_is_broadcasted = (in_is_broadcasted || mask_is_broadcasted);

  size_t out_ix = 0;
  for (size_t i = 0; i < broadcast_numel; ++i) {
    size_t in_linear_index = i;
    size_t mask_linear_index = i;

    // If either `in` or `mask` is broadcasted, we need to compute the indexes
    // in the broadcasted space.
    if (any_is_broadcasted) {
      size_t broadcast_indexes[kTensorDimensionLimit];
      delinearize_index(
          i,
          {broadcast_sizes, broadcast_ndim},
          broadcast_indexes,
          kTensorDimensionLimit);

      if (in_is_broadcasted) {
        in_linear_index =
            linearize_access_indexes(broadcast_indexes, broadcast_ndim, in);
      }
      if (mask_is_broadcasted) {
        mask_linear_index =
            linearize_access_indexes(broadcast_indexes, broadcast_ndim, mask);
      }
    }

    // If the mask is true, copy the value from `in` to `out` and increment the
    // `out_ix`
    if (mask_data[mask_linear_index]) {
      memcpy(
          out_data + out_ix * elem_size,
          in_data + in_linear_index * elem_size,
          elem_size);
      out_ix++;
    }
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

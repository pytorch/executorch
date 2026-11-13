/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

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

  const size_t broadcast_numel = compute_numel(broadcast_sizes, broadcast_ndim);

  // Compute the number of out elements
  size_t mask_true_count = 0;
  const bool* const mask_data = mask.const_data_ptr<bool>();
  for (const auto i : c10::irange(mask.numel())) {
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

  size_t out_ix = 0;
  // Keep separate copy bodies to avoid per-element calls in GCC -Os builds.
  if (in.sizes() == mask.sizes()) {
    for (const auto i : c10::irange(broadcast_numel)) {
      if (mask_data[i]) {
        memcpy(
            out_data + out_ix * elem_size, in_data + i * elem_size, elem_size);
        out_ix++;
      }
    }
  } else {
    for (const auto [unused_index, in_index, mask_index] :
         BroadcastIndexesRange<2>(
             {broadcast_sizes, broadcast_ndim}, in, mask)) {
      (void)unused_index;
      if (mask_data[mask_index]) {
        memcpy(
            out_data + out_ix * elem_size,
            in_data + in_index * elem_size,
            elem_size);
        out_ix++;
      }
    }
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

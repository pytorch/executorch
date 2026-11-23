/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_indexes_range.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& masked_scatter_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& mask,
    const Tensor& src,
    Tensor& out) {
  ScalarType in_type = in.scalar_type();

  ET_KERNEL_CHECK(
      ctx,
      executorch::runtime::tensor_is_realhbbf16_type(in),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, mask.scalar_type() == ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, src.scalar_type() == in_type, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, out.scalar_type() == in_type, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, mask, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(in, mask, out) == Error::Ok,
      InvalidArgument,
      out);

  int64_t idx = 0;
  int64_t src_numel = src.numel();
  bool src_numel_check = true;

  static constexpr auto name = "masked_scatter.out";

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, name, CTYPE, [&]() {
    const CTYPE* const src_data = src.const_data_ptr<CTYPE>();
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    const bool* const mask_data = mask.const_data_ptr<bool>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    for (const auto [unused_index, in_index, mask_index, out_index] :
         BroadcastIndexesShapeRange<
             3,
             /*support_noncontiguous_input_tensors=*/true>(
             out.sizes(), in, mask, out)) {
      if (mask_data[mask_index]) {
        if (idx >= src_numel) {
          src_numel_check = false;
          out_data[out_index] = in_data[in_index];
        } else {
          out_data[out_index] = src_data[idx++];
        }
      } else {
        out_data[out_index] = in_data[in_index];
      }
    }
  });

  ET_KERNEL_CHECK_MSG(
      ctx,
      src_numel_check,
      InvalidArgument,
      out,
      "masked_scatter: src doesn't have enough elements");

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

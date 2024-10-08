/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/slice_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

Tensor& slice_scatter_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& src,
    int64_t dim,
    exec_aten::optional<int64_t> start_val,
    exec_aten::optional<int64_t> end_val,
    int64_t step,
    Tensor& out) {
  (void)ctx;

  if (dim < 0) {
    dim += input.dim();
  }

  // resize out tensor for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, input.sizes()) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(input, out), InvalidArgument, out);

  if (input.numel() == 0) {
    return out;
  }

  ET_KERNEL_CHECK(ctx, dim >= 0 && dim < input.dim(), InvalidArgument, out);

  // If user do not set value to end_val, set end to input.size(dim) (largest
  // value available)
  int64_t end = end_val.has_value() ? end_val.value() : input.size(dim);
  // If user do not set value to start_val, set start to 0 (smallest value
  // available)
  int64_t start = start_val.has_value() ? start_val.value() : 0;

  ET_KERNEL_CHECK(ctx, step > 0, InvalidArgument, out);

  int64_t num_values =
      adjust_slice_indices(input.size(dim), &start, &end, step);

  ET_KERNEL_CHECK(
      ctx,
      check_slice_scatter_args(input, src, dim, num_values, step, out),
      InvalidArgument,
      out);

  size_t dim_length = input.size(dim);
  size_t leading_dims = getLeadingDims(input, dim);
  size_t trailing_dims = getTrailingDims(input, dim);

  // To start, copy the input into the output
  memcpy(out.mutable_data_ptr(), input.const_data_ptr(), input.nbytes());

  ScalarType in_type = input.scalar_type();
  ScalarType src_type = src.scalar_type();

  ET_SWITCH_REALHBBF16_TYPES(in_type, ctx, "slice_scatter.out", CTYPE, [&]() {
    ET_SWITCH_REALHBBF16_TYPES(
        src_type, ctx, "slice_scatter.out", CTYPE_SRC, [&]() {
          CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
          const CTYPE_SRC* src_data = src.const_data_ptr<CTYPE_SRC>();

          size_t src_offset = 0;

          for (int i = 0; i < leading_dims; i++) {
            size_t out_offset = (i * dim_length + start) * trailing_dims;
            for (int j = 0; j < num_values; j++) {
              for (size_t k = 0; k < trailing_dims; ++k) {
                out_data[out_offset + k] =
                    convert<CTYPE, CTYPE_SRC>(src_data[src_offset + k]);
              }
              src_offset += trailing_dims;
              out_offset += step * trailing_dims;
            }
          }
        });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

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

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

/// aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *,
/// Tensor(a!) out) -> Tensor(a!)
Tensor& select_scatter_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, src, out), InvalidArgument, out);

  // Account for negative indices
  if (dim < 0) {
    dim += in.dim();
  }

  ET_KERNEL_CHECK(ctx, dim >= 0 && dim < in.dim(), InvalidArgument, out);

  if (index < 0) {
    index += in.size(dim);
  }

  // Check args
  ET_KERNEL_CHECK(
      ctx,
      check_select_scatter_args(in, src, dim, index, out),
      InvalidArgument,
      out);

  // If the input is an empty tensor, no other operation could be done. We just
  // return the output.
  if (in.numel() == 0) {
    return out;
  }

  // To start, copy the input into the output. Input will not be empty due to
  // the checks performed above.
  memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());

  // Strides to help with memory address arithmetic
  size_t leading_dims = getLeadingDims(in, dim);
  size_t trailing_stride = getTrailingDims(in, dim);
  size_t start_offset = index * trailing_stride;
  size_t out_step = in.size(dim) * trailing_stride;

  ScalarType in_type = in.scalar_type();
  ScalarType src_type = src.scalar_type();

  ET_SWITCH_REAL_TYPES_AND(
      Bool, in_type, ctx, "select_scatter.out", CTYPE, [&]() {
        ET_SWITCH_REAL_TYPES_AND(
            Bool, src_type, ctx, "select_scatter.out", CTYPE_SRC, [&]() {
              CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();
              const CTYPE_SRC* const src_data = src.const_data_ptr<CTYPE_SRC>();

              for (size_t i = 0; i < leading_dims; ++i) {
                for (size_t j = 0; j < trailing_stride; ++j) {
                  out_data[start_offset + i * out_step + j] =
                      convert<CTYPE, CTYPE_SRC>(
                          src_data[i * trailing_stride + j]);
                }
              }
            });
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

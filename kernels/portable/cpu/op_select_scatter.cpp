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

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

/**
 * Assumptions for inputs:
 * 1. output size is the same as input size
 * 2. src size is the same as the selected slice from the input
 * 3. dim and index values are valid given the input tensor
 */
void check_select_scatter_args(
    const Tensor& in,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& output) {
  // Support python-style negative indexing. E.g., for the shape {2, 3, 4},
  // dim = -1 would refer to dim[2], dim = -2 would refer to dim[1], and so on.

  // The dim planed to be selected on shall exist in input
  ET_CHECK_MSG(
      dim >= 0 && dim < in.dim(),
      "dim %" PRId64 " out of range [-%zd,%zd)",
      dim,
      in.dim(),
      in.dim());

  // The index shall be valid in the given dimenson
  ET_CHECK_MSG(
      index >= 0 && index < in.size(dim),
      "index %" PRId64 " out of range [-%zd,%zd) at in.size( %" PRId64 ")",
      index,
      in.size(dim),
      in.size(dim),
      dim);

  // The src.dim() shall be one lower than in.dim() since src needs to fit
  // into the selected data on one dim of input
  // https://pytorch.org/docs/stable/generated/torch.select_scatter.html
  ET_CHECK_MSG(
      in.dim() == src.dim() + 1,
      "in.dim() %zd != src.dim() + 1 %zd",
      in.dim(),
      src.dim() + 1);

  // The size of src tensor should follow these rules:
  // - src.size(i) shall equal to in.size(i) if i < dim,
  // - src.size(i) shall equal to in.size(i+1) if i >= dim

  for (ssize_t d = 0; d < in.dim() - 1; d++) {
    if (d < dim) {
      ET_CHECK_MSG(
          in.size(d) == src.size(d),
          "in.size(%zu) %zd != src.size(%zu) %zd | dim = %" PRId64 ")",
          d,
          in.size(d),
          d,
          src.size(d),
          dim);
    } else {
      ET_CHECK_MSG(
          in.size(d + 1) == src.size(d),
          "in.size(%zu) %zd != src.size(%zu) %zd | dim = %" PRId64 ")",
          d + 1,
          in.size(d + 1),
          d,
          src.size(d),
          dim);
    }
  }
}

} // namespace

/// aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *,
/// Tensor(a!) out) -> Tensor(a!)
Tensor& select_scatter_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& src,
    int64_t dim,
    int64_t index,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(ctx, tensors_have_same_dtype(in, out), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);

  // Account for negative indices
  if (dim < 0) {
    dim += in.dim();
  }
  if (index < 0) {
    index += in.size(dim);
  }

  // Check args
  check_select_scatter_args(in, src, dim, index, out);

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

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

namespace {
void check_args(const Tensor& input, int64_t dim, TensorList out) {
  ET_CHECK_MSG(
      input.dim() > 0,
      "input must have at least one dimension; saw %zd",
      input.dim());
  ET_CHECK_MSG(
      dim >= 0 && dim < input.dim(),
      "dim %" PRId64 " out of range [0,%zd)",
      dim,
      input.dim());

  const ssize_t dim_size = input.size(dim);
  ET_CHECK_MSG(
      dim_size == out.size(),
      "out tensorlist's length %zd must equal unbind dim %" PRId64
      " size = %zd.",
      out.size(),
      dim,
      dim_size);

  // Validate each output.
  for (size_t i = 0; i < out.size(); ++i) {
    // All output dtypes must be the same.
    ET_CHECK_MSG(
        out[i].scalar_type() == out[0].scalar_type(),
        "out[%zu] dtype %hhd != out[0] dtype %hhd",
        i,
        out[i].scalar_type(),
        out[0].scalar_type());

    // output tensor must have # of dims = input.dim() -1
    ET_CHECK_MSG(
        out[i].dim() == (input.dim() - 1),
        "out[%zu] dim %zd != input dim %zd",
        i,
        out[i].dim(),
        input.dim() - 1);

    // Check the shape of the output.
    for (ssize_t d = 0, out_d = 0; d < input.dim(); ++d) {
      if (d != dim) {
        ET_CHECK_MSG(
            out[i].size(out_d) == input.size(d),
            "out[%zu].size(%zd) %zd != input.size(%zd) %zd",
            i,
            d,
            out[i].size(out_d),
            d,
            input.size(d));
        out_d++;
      }
    }
  }
}

} // namespace

/**
 * unbind_copy.int_out(Tensor input, int dim=0, *, Tensor(a!)[] out) -> ()
 */
void unbind_copy_int_out(
    RuntimeContext& ctx,
    const Tensor& input,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing.
  if (dim < 0) {
    dim += input.dim();
  }
  check_args(input, dim, out);

  if (input.numel() == 0) {
    return;
  }

  const size_t leading_dims = getLeadingDims(input, dim);
  const size_t trailing_dims = getTrailingDims(input, dim);
  const size_t step = input.size(dim) * trailing_dims;

  ScalarType in_type = input.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  ET_SWITCH_REAL_TYPES_AND(Bool, in_type, ctx, __func__, CTYPE_IN, [&]() {
    ET_SWITCH_REAL_TYPES_AND(Bool, out_type, ctx, __func__, CTYPE_OUT, [&]() {
      const CTYPE_IN* const input_data = input.const_data_ptr<CTYPE_IN>();
      for (size_t i = 0, e = out.size(); i < e; ++i) {
        size_t input_offset = i * trailing_dims;
        CTYPE_OUT* const dest = out[i].mutable_data_ptr<CTYPE_OUT>();
        size_t dest_offset = 0;
        for (size_t j = 0; j < leading_dims; ++j) {
          for (size_t k = 0; k < trailing_dims; ++k) {
            dest[dest_offset + k] =
                convert<CTYPE_OUT, CTYPE_IN>(input_data[input_offset + k]);
          }
          input_offset += step;
          dest_offset += trailing_dims;
        }
      }
    });
  });
}

} // namespace native
} // namespace executor
} // namespace torch

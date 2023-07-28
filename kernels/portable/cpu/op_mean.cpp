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

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

namespace {

void check_preconditions(
    const Tensor& in,
    const optional<ArrayRef<int64_t>>& dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  const ScalarType out_dtype = out.scalar_type();
  const ScalarType in_dtype = in.scalar_type();
  if (dtype.has_value()) {
    ET_CHECK_MSG(
        dtype.value() == ScalarType::Float ||
            dtype.value() == ScalarType::Double,
        "dtype must be a floating point dtype");
    ET_CHECK_MSG(
        dtype.value() == out_dtype,
        "out tensor should be of the same dtype with dtype");
  } else {
    ET_CHECK_MSG(
        in_dtype == ScalarType::Float || in_dtype == ScalarType::Double,
        "in tensor must have a floating point dtype");
    ET_CHECK_MSG(
        out_dtype == ScalarType::Float || out_dtype == ScalarType::Double,
        "out tensor must have a floating point dtype");
  }
  check_dim_list_is_valid(in, dim_list);
  ET_CHECK_MSG(
      out.dim() == compute_reduced_out_dim(in, dim_list, keepdim),
      "Number of dims of out tensor is not compatible with inputs and params");
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);
}

} // namespace

Tensor& mean_dim_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  check_preconditions(in, dim_list, keepdim, dtype, out);

  Error e = resize_reduction_out(in, dim_list, keepdim, out);
  ET_CHECK_MSG(e == Error::Ok, "Failed to resize out tensor in mean_dim_out");

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "mean", CTYPE_IN, [&] {
    ET_SWITCH_FLOAT_TYPES(out.scalar_type(), ctx, "mean", CTYPE_OUT, [&] {
      CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
      const size_t num = get_reduced_dim_product(in, dim_list);
      for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
        CTYPE_OUT sum = 0;
        if (in.numel() > 0) {
          sum = map_reduce_over_dim_list<CTYPE_IN, CTYPE_OUT>(
              [](CTYPE_IN v) { return static_cast<CTYPE_OUT>(v); },
              [](CTYPE_OUT outv, CTYPE_OUT acc) { return acc + outv; },
              in,
              dim_list,
              out_ix);
        }
        out_data[out_ix] = sum / num;
      }
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

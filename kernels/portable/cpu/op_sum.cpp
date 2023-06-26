// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <executorch/core/Assert.h>
#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>

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
  if (dtype.has_value()) {
    ET_CHECK_MSG(
        dtype.value() == out.scalar_type(),
        "out tensor should be of the same dtype with dtype");
  }
  check_dim_list_is_valid(in, dim_list);
  ET_CHECK_MSG(
      out.dim() == compute_reduced_out_dim(in, dim_list, keepdim),
      "Number of dims of out tensor is not compatible with inputs and params");
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);
}

} // namespace

Tensor& sum_dim_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  check_preconditions(in, dim_list, keepdim, dtype, out);

  Error e = resize_reduction_out(in, dim_list, keepdim, out);
  ET_CHECK_MSG(e == Error::Ok, "Failed to resize out tensor in sum_dim_out");

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "sum", CTYPE_IN, [&] {
    ET_SWITCH_REAL_TYPES_AND(
        Bool, out.scalar_type(), ctx, "sum", CTYPE_OUT, [&] {
          CTYPE_OUT* out_data = out.mutable_data_ptr<CTYPE_OUT>();
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
            out_data[out_ix] = sum;
          }
        });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>

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
    const ArrayRef<int64_t>& dim_list,
    bool keepdim,
    Tensor& out) {
  ET_CHECK_SAME_DTYPE2(in, out);
  check_dim_list_is_valid(in, dim_list);
  for (const auto& d : dim_list) {
    ET_CHECK_NON_ZERO_DIM_SIZE(d, in);
  }
  ET_CHECK_MSG(
      out.dim() == compute_reduced_out_dim(in, dim_list, keepdim),
      "Number of dims of out tensor is not compatible with inputs and params");
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);
}

} // namespace

Tensor& amin_out(
    RuntimeContext& ctx,
    const Tensor& in,
    ArrayRef<int64_t> dim_list,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  check_preconditions(in, dim_list, keepdim, out);

  Error e = resize_reduction_out(in, dim_list, keepdim, out);
  ET_CHECK_MSG(e == Error::Ok, "Failed to resize out tensor in amin_out");

  ET_SWITCH_REAL_TYPES_AND(Bool, in.scalar_type(), ctx, "amin", CTYPE, [&]() {
    CTYPE* out_data = out.mutable_data_ptr<CTYPE>();
    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      out_data[out_ix] = reduce_over_dim_list<CTYPE>(
          [](CTYPE v, CTYPE min_v) {
            return std::isnan(v) || v < min_v ? v : min_v;
          },
          in,
          dim_list,
          out_ix);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cmath>

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using exec_aten::optional;
using exec_aten::Tensor;

namespace {

void check_preconditions(
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  if (dim.has_value()) {
    ET_CHECK_VALID_DIM(dim.value(), in.dim());
    ET_CHECK_NON_ZERO_DIM_SIZE(dim.value(), in);
  }
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Long,
      "Expected out tensor to have dtype Long, but got %hhd instead",
      out.scalar_type());
  ET_CHECK_MSG(
      out.dim() == compute_reduced_out_dim(in, dim, keepdim),
      "Number of dims of out tensor is not compatible with inputs and params");
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);
}

} // namespace

Tensor& argmax_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  check_preconditions(in, dim, keepdim, out);

  Error error = resize_reduction_out(in, dim, keepdim, out);
  ET_CHECK_MSG(error == Error::Ok, "Failed to resize out tensor in argmax_out");

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "argmax", CTYPE, [&] {
    long* out_data = out.mutable_data_ptr<long>();

    for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
      std::tuple<CTYPE, long> acc = reduce_over_dim<CTYPE>(
          [](CTYPE v, long ix, CTYPE acc_val, long acc_ix) {
            if (!std::isnan(acc_val) && (std::isnan(v) || v > acc_val)) {
              acc_val = v;
              acc_ix = ix;
            }
            return std::tuple<CTYPE, long>{acc_val, acc_ix};
          },
          in,
          dim,
          out_ix);
      out_data[out_ix] = std::get<1>(acc);
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch

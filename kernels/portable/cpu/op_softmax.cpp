// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cmath>

#include <executorch/kernels/kernel_includes.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {

void check_preconditions(
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  // Ensure half_to_float is not true
  ET_CHECK_MSG(
      !half_to_float,
      "softmax with half to float conversion is not supported on CPU");
  // Check both in and out are of the same dtype
  ET_CHECK_SAME_DTYPE2(in, out);
  // Check both in and out have the same number of dimensions
  ET_CHECK_MSG(
      in.dim() == out.dim(),
      "in.dim() %zd!= out.dim() %zd",
      in.dim(),
      out.dim());
  // Ensure in has value
  ET_CHECK_MSG(in.numel() > 0, "in.numel() %zd <= 0", in.numel());
  // Ensure dim is valid
  ET_CHECK_VALID_DIM(dim, in.dim());
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);
}

} // namespace

Tensor& softmax_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool half_to_float,
    Tensor& out) {
  (void)ctx;

  check_preconditions(in, dim, half_to_float, out);

  Error err = resize_tensor(out, in.sizes());
  ET_CHECK_MSG(err == Error::Ok, "Failed to resize out tensor in softmax_out");

  // Adjust for negative dim
  dim = dim < 0 ? dim + in.dim() : dim;

  ET_SWITCH_FLOAT_TYPES(in.scalar_type(), ctx, "softmax", CTYPE, [&]() {
    const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
    CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

    apply_over_dim(
        [in_data, out_data](
            const size_t size, const size_t stride, const size_t base) {
          // calculate max in softmax dim. During softmax computation each
          // value is subtracted by the maximum in value before calling exp
          // to preserve numerical stability.
          const CTYPE max_in = apply_unary_reduce_fn(
              [](const CTYPE val_in, CTYPE val_accum) {
                return std::max(val_in, val_accum);
              },
              in_data + base,
              size,
              stride);

          const CTYPE temp_sum = apply_unary_map_reduce_fn<CTYPE, CTYPE>(
              [max_in](const CTYPE val_in) {
                return std::exp(val_in - max_in);
              },
              [](const CTYPE mapped_in, CTYPE val_accum) {
                return val_accum + mapped_in;
              },
              in_data + base,
              size,
              stride);

          apply_unary_map_fn(
              [max_in, temp_sum](const CTYPE val_in) {
                return std::exp(val_in - max_in) / temp_sum;
              },
              in_data + base,
              out_data + base,
              size,
              stride);
        },
        in,
        dim);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
